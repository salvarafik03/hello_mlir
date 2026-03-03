#include "Melo/MeloDialect.h"
#include "Melo/MeloOps.h"
#include "Melo/MeloPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace melo;

namespace {

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
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<tosa::TosaDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AbsOpLowering, AddOpLowering>(&getContext());

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
