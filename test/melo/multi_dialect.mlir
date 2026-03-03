// Multi-dialect test in one function: Melo + TOSA + Arith.
module {
  func.func @multi_dialect_in_one_func(
      %a: tensor<4xf32>, %b: tensor<4xf32>, %s: f32) -> (tensor<4xf32>, f32) {
    %0 = melo.add %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %1 = tosa.abs %0 : (tensor<4xf32>) -> tensor<4xf32>
    %2 = arith.addf %s, %s : f32
    return %1, %2 : tensor<4xf32>, f32
  }
}
