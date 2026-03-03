module {
  func.func @test_abs_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = melo.abs %arg0 : tensor<4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
