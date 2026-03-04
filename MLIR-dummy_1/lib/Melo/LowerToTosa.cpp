#include "Melo/MeloDialect.h"
#include "Melo/MeloOps.h"
#include "Melo/MeloPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace melo;

namespace {

static FailureOr<int32_t> normalizeAxis(int64_t axis, int64_t rank) {
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return failure();
  return static_cast<int32_t>(axis);
}

/// Lower melo.abs -> tosa.abs.
class AbsOpLowering : public OpConversionPattern<AbsOp> {
public:
  using OpConversionPattern<AbsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = op.getType();
    Value input = adaptor.getInput();

    if (isa<TensorType>(resultType)) {
      rewriter.replaceOpWithNewOp<tosa::AbsOp>(op, resultType, input);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported type for melo.abs");
  }
};

/// Lower melo.add -> tosa.add.
class AddOpLowering : public OpConversionPattern<AddOp> {
public:
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type resultType = op.getType();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (isa<TensorType>(resultType)) {
      rewriter.replaceOpWithNewOp<tosa::AddOp>(op, resultType, lhs, rhs);
      return success();
    }

    if (isa<FloatType>(resultType)) {
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, resultType, lhs, rhs);
      return success();
    }

    if (isa<IntegerType>(resultType)) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, resultType, lhs, rhs);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported type for melo.add");
  }
};

/// Lower melo.softmax -> TOSA primitive op decomposition.
class SoftmaxOpLowering : public OpConversionPattern<SoftmaxOp> {
public:
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "requires ranked tensor types");

    if (!isa<FloatType>(inputType.getElementType()))
      return rewriter.notifyMatchFailure(op, "only float softmax is supported");

    FailureOr<int32_t> normalizedAxis =
        normalizeAxis(op.getDim(), inputType.getRank());
    if (failed(normalizedAxis))
      return rewriter.notifyMatchFailure(op, "invalid softmax dim");

    SmallVector<int64_t> reducedShape(inputType.getShape().begin(),
                                      inputType.getShape().end());
    reducedShape[*normalizedAxis] = 1;
    auto reducedType =
        RankedTensorType::get(reducedShape, inputType.getElementType());

    auto axisAttr = rewriter.getI32IntegerAttr(*normalizedAxis);
    Location loc = op.getLoc();

    // Stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    Value reducedMax = rewriter
                           .create<tosa::ReduceMaxOp>(loc, reducedType,
                                                      adaptor.getInput(), axisAttr)
                           .getOutput();
    Value shifted = rewriter
                        .create<tosa::SubOp>(loc, inputType, adaptor.getInput(),
                                             reducedMax)
                        .getOutput();
    Value exps = rewriter.create<tosa::ExpOp>(loc, inputType, shifted).getOutput();
    Value denom = rewriter
                      .create<tosa::ReduceSumOp>(loc, reducedType, exps, axisAttr)
                      .getOutput();
    Value invDenom =
        rewriter.create<tosa::ReciprocalOp>(loc, reducedType, denom).getOutput();

    auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
    auto shiftAttr = DenseIntElementsAttr::get(shiftType, {0});
    Value shift =
        rewriter.create<arith::ConstantOp>(loc, shiftType, shiftAttr).getResult();

    Value output = rewriter
                       .create<tosa::MulOp>(loc, resultType, exps, invDenom, shift)
                       .getOutput();
    rewriter.replaceOp(op, output);
    return success();
  }
};

struct LowerToTosaPass
    : public PassWrapper<LowerToTosaPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToTosaPass)

  StringRef getArgument() const final { return "melo-lower-to-tosa"; }

  StringRef getDescription() const final {
    return "Lower Melo dialect operations to TOSA dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ConversionTarget target(getContext());
    target.addIllegalDialect<melo::MeloDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<tosa::TosaDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AbsOpLowering, AddOpLowering, SoftmaxOpLowering>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace melo {

std::unique_ptr<Pass> createLowerToTosaPass() {
  return std::make_unique<LowerToTosaPass>();
}

void registerMeloPasses() {
  static PassRegistration<LowerToTosaPass> pass;
}

} // namespace melo
