// Multi-dialect test in one function: Arith + TOSA + SCF.
module {
  func.func @multi_dialect_example(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %cond = arith.constant true
    %abs = tosa.abs %arg0 : (tensor<4xf32>) -> tensor<4xf32>
    %add = tosa.add %abs, %abs : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

    %0 = scf.if %cond -> (tensor<4xf32>) {
      scf.yield %add : tensor<4xf32>
    } else {
      scf.yield %abs : tensor<4xf32>
    }
    return %0 : tensor<4xf32>
  }
}
