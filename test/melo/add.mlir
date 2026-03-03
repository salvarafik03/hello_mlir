// Test addition operation for scalars
func.func @test_add_scalars(%arg0: f32, %arg1: f32) -> f32 {
  %0 = melo.add %arg0, %arg1 : f32, f32 -> f32
  return %0 : f32
}

// Test addition operation for tensors
func.func @test_add_tensors(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = melo.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
