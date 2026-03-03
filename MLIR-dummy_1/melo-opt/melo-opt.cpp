#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Melo/MeloDialect.h"
#include "Melo/MeloPasses.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input melo file>"),
    cl::init("-"),
    cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int processMLIR(mlir::MLIRContext &context,
                mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::PassManager pm(&context);

  // Add Melo → TOSA lowering pass
  pm.addPass(melo::createLowerToTosaPass());

  if (mlir::failed(pm.run(module.get()))) {
    llvm::errs() << "Failed to run Melo → TOSA lowering\n";
    return 1;
  }

  llvm::outs() << *module;
  return 0;
}

int main(int argc, char **argv) {
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Melo compiler\n");

  mlir::MLIRContext context;

  // Register dialects
  context.getOrLoadDialect<melo::MeloDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  // Register passes
  melo::registerMeloPasses();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadMLIR(context, module))
    return error;

  if (int error = processMLIR(context, module))
    return error;

  return 0;
}
