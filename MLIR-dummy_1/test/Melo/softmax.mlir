// RUN: melo-opt %s | FileCheck %s

module {
  func.func @test_softmax_dim_last(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = melo.softmax %arg0 dim = 1 : tensor<2x4xf32> -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }

  func.func @test_softmax_dim_negative(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    %0 = melo.softmax %arg0 dim = -1 : tensor<2x3x4xf32> -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}

// CHECK: func.func @test_softmax_dim_last(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
// CHECK:   %[[MAX:.+]] = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<2x4xf32>) -> tensor<2x1xf32>
// CHECK:   %[[SHIFTED:.+]] = tosa.sub %arg0, %[[MAX]] : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
// CHECK:   %[[EXP:.+]] = tosa.exp %[[SHIFTED]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK:   %[[SUM:.+]] = tosa.reduce_sum %[[EXP]] {axis = 1 : i32} : (tensor<2x4xf32>) -> tensor<2x1xf32>
// CHECK:   %[[INV:.+]] = tosa.reciprocal %[[SUM]] : (tensor<2x1xf32>) -> tensor<2x1xf32>
// CHECK:   %[[SHIFT:.+]] = arith.constant dense<0> : tensor<1xi8>
// CHECK:   %[[OUT:.+]] = tosa.mul %[[EXP]], %[[INV]], %[[SHIFT]] : (tensor<2x4xf32>, tensor<2x1xf32>, tensor<1xi8>) -> tensor<2x4xf32>
// CHECK:   return %[[OUT]] : tensor<2x4xf32>
// CHECK: }

// CHECK: func.func @test_softmax_dim_negative(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
// CHECK:   %[[MAX2:.+]] = tosa.reduce_max %arg0 {axis = 2 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
// CHECK:   %[[SHIFTED2:.+]] = tosa.sub %arg0, %[[MAX2]] : (tensor<2x3x4xf32>, tensor<2x3x1xf32>) -> tensor<2x3x4xf32>
// CHECK:   %[[EXP2:.+]] = tosa.exp %[[SHIFTED2]] : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK:   %[[SUM2:.+]] = tosa.reduce_sum %[[EXP2]] {axis = 2 : i32} : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
// CHECK:   %[[INV2:.+]] = tosa.reciprocal %[[SUM2]] : (tensor<2x3x1xf32>) -> tensor<2x3x1xf32>
// CHECK:   %[[SHIFT2:.+]] = arith.constant dense<0> : tensor<1xi8>
// CHECK:   %[[OUT2:.+]] = tosa.mul %[[EXP2]], %[[INV2]], %[[SHIFT2]] : (tensor<2x3x4xf32>, tensor<2x3x1xf32>, tensor<1xi8>) -> tensor<2x3x4xf32>
// CHECK:   return %[[OUT2]] : tensor<2x3x4xf32>
// CHECK: }
