// RUN: melo-opt %s | FileCheck %s

// Test absolute value operation for scalar
func.func @test_abs_scalar(%arg0: f32) -> f32 {
  %0 = melo.abs %arg0 : f32 -> f32
  return %0 : f32
}

// CHECK: func.func @test_abs_scalar(%arg0: f32) -> f32 {
// CHECK:   %0 = melo.abs %arg0 : f32 -> f32
// CHECK:   return %0 : f32
// CHECK: }

// Test absolute value operation for tensor
func.func @test_abs_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = melo.abs %arg0 : tensor<4xf32> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func.func @test_abs_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:   %0 = melo.abs %arg0 : tensor<4xf32> -> tensor<4xf32>
// CHECK:   return %0 : tensor<4xf32>
// CHECK: }