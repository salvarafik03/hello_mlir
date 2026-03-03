// Mixed dialect test in one function: Melo + TOSA.
module {
  func.func @mixed_melo_tosa(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = melo.add %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %1 = tosa.abs %0 : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
