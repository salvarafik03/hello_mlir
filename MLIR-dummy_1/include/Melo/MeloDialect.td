#ifndef MELO_DIALECT
#define MELO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Melo dialect definition.
//===----------------------------------------------------------------------===//

def Melo_Dialect : Dialect {
    let name = "melo";
    let summary = "A melo out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement melo-world kind of sample code
        for MLIR.
    }];
    let cppNamespace = "::melo";
}

//===----------------------------------------------------------------------===//
// Base melo operation definition.
//===----------------------------------------------------------------------===//

class Melo_Op<string mnemonic, list<Trait> traits = []> :
        Op<Melo_Dialect, mnemonic, traits>;

#endif // MELO_DIALECT
