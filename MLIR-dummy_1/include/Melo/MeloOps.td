#ifndef MELO_OPS
#define MELO_OPS

include "MeloDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ---------------------------------------------
// Minimal dummy operation for build testing
// ---------------------------------------------

def MeloDummyOp : Melo_Op<"dummy", [Pure]> {
  let summary = "Dummy operation for testing build";
  let description = [{
    Minimal operation just to verify the dialect build pipeline.
  }];

  let arguments = (ins AnyType:$input);
  let results   = (outs AnyType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// ---------------------------------------------
// Absolute value operation
// ---------------------------------------------

def Melo_AbsOp : Melo_Op<"abs", [Pure]> {
  let summary = "Absolute value operation";
  let description = [{
    Computes the absolute value of the input.
    Supports scalar and tensor types.
  }];

  let arguments = (ins AnyType:$input);
  let results   = (outs AnyType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

// ---------------------------------------------
// Addition operation
// ---------------------------------------------

def Melo_AddOp : Melo_Op<"add", [Pure]> {
  let summary = "Addition operation";
  let description = [{
    Adds two inputs.
    Supports scalar and tensor types.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$output);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

#endif // MELO_OPS