/**
 * @file minimal_swiglu.cpp
 * @brief Implementation of SwiGLU (Swish-Gated Linear Unit) operations using XNNPACK
 * 
 * This file implements the SwiGLU activation function, which is a variant of GLU that uses
 * SiLU (Swish) as the gating function. The implementation supports various quantization
 * schemes (fp32, 8-bit, 4-bit) and includes runtime caching for better performance.
 * 
 * The main computation flow is:
 * output = W2 @ (SiLU(W1 @ input) * (W3 @ input))
 * where @ denotes matrix multiplication and * denotes element-wise multiplication
 */
#include <stdio.h>
#include <math.h>
#include <xnnpack.h>
#include <vector>


// Use #define for compile-time constants to allow array initialization
#define INPUT_DIM  3
#define OUTPUT_DIM 2
#define INTER_DIM 4
#define BATCH_SIZE 1

int main(void) {
  // 1. Initialize XNNPACK
  if (xnn_initialize(NULL) != xnn_status_success) {
    fprintf(stderr, "Failed to initialize XNNPACK\n");
    return 1;
  }

  const size_t input_dim  = INPUT_DIM;
  const size_t output_dim = OUTPUT_DIM;
  const size_t inter_dim = INTER_DIM;


    // Weights are in row-major order. We will reuse w1_weights for w3.
    float w1_weight_data[INTER_DIM * INPUT_DIM];
    for (size_t i = 0; i < INTER_DIM; ++i) {
        for (size_t j = 0; j < INPUT_DIM; ++j) {
            w1_weight_data[i * INPUT_DIM + j] = static_cast<float>(i * INPUT_DIM + j + 1) / (INTER_DIM * INPUT_DIM);
        }
    }

    float w2_weight_data[OUTPUT_DIM * INTER_DIM];
    for (size_t i = 0; i < OUTPUT_DIM; ++i) {
        for (size_t j = 0; j < INTER_DIM; ++j) {
            w2_weight_data[i * INTER_DIM + j] = static_cast<float>(i * INTER_DIM + j + 1) / (OUTPUT_DIM * INTER_DIM);
        }
    }

  xnn_subgraph_t subgraph = NULL;
  enum xnn_status status = xnn_create_subgraph(
    /*external_value_ids=*/2,  // we have 2 external values: input and output
    /*flags=*/0,
    &subgraph);
  if (status != xnn_status_success) {
    fprintf(stderr, "xnn_create_subgraph failed: %d\n", status);
    return 1;
  }

    // Define input tensor
    uint32_t input_id;
    {
        std::vector<size_t> input_dims = {1, INPUT_DIM};
        status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            /*num_dims=*/input_dims.size(),
            /*dims=*/input_dims.data(),
            /*data=*/nullptr,  // Data will be provided during setup
            /*external_id=*/0,  // Input external ID
            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,  // Mark as external input
            &input_id);
        if (status != xnn_status_success) {
            fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
            return 1;
        }
    }

    // Define output tensor
    uint32_t output_id;
    {
        std::vector<size_t> output_dims = {1, OUTPUT_DIM};
        status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            /*num_dims=*/output_dims.size(),
            /*dims=*/output_dims.data(),
            /*data=*/nullptr,  // Data will be provided during setup
            /*external_id=*/1,  // Output external ID
            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,  // Mark as external output
            &output_id);
        if (status != xnn_status_success) {
            fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
            return 1;
        }
    }

    // Gate projection: w1 @ input
    // Define w1 weight tensor (gate projection)
  
    uint32_t w1_weight_id;
    {
        std::vector<size_t> w1_weight_dims = {inter_dim, input_dim};
        status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            /*num_dims=*/w1_weight_dims.size(),
            /*dims=*/w1_weight_dims.data(),
            /*data=*/w1_weight_data,
            /*external_id=*/XNN_INVALID_VALUE_ID,
            /*flags=*/0,
            &w1_weight_id);
        if (status != xnn_status_success) {
            fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
            return 1;
        }
    }

    // Gate projection output (w1 @ input)
    uint32_t gate_output_id;
    {
        std::vector<size_t> gate_dims = {1, inter_dim}; // will reshape later to match batch size
        status = xnn_define_tensor_value(
            subgraph,
            xnn_datatype_fp32,
            /*num_dims=*/gate_dims.size(),
            /*dims=*/gate_dims.data(),
            /*data=*/nullptr,
            /*external_id=*/XNN_INVALID_VALUE_ID,
            /*flags=*/0,
            &gate_output_id);
        if (status != xnn_status_success) {
            fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
            return 1;
        }
    }


  // Define gate projection operation
  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-INFINITY,
      /*output_max=*/INFINITY,
      /*input_id=*/input_id,
      /*filter_id=*/w1_weight_id,
      /*bias_id=*/XNN_INVALID_VALUE_ID,  // No bias
      /*output_id=*/gate_output_id,
      /*flags=*/0);

  if (status != xnn_status_success) {
    fprintf(stderr, "xnn_define_fully_connected failed: %d\n", status);
    return 1;
  }


  // Up projection: w3 @ input
  uint32_t w3_weight_id;
  {
    std::vector<size_t> w3_weight_dims = {inter_dim, input_dim};
    status = xnn_define_tensor_value(
      subgraph,
      xnn_datatype_fp32,
      /*num_dims=*/w3_weight_dims.size(),
      /*dims=*/w3_weight_dims.data(),
      /*data=*/w1_weight_data, // using the same weights for up projection as the gate projection
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0,
      &w3_weight_id);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
      return 1;
    }
  }

    // Up projection output (w3 @ input)
    uint32_t up_output_id;
    {
      std::vector<size_t> up_output_dims = {1, inter_dim}; // will reshape later to match batch size
      status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        /*num_dims=*/up_output_dims.size(),
        /*dims=*/up_output_dims.data(),
        /*data=*/nullptr,
        /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0,
        &up_output_id);
      if (status != xnn_status_success) {
        fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
        return 1;
      }
    }

      // Define up projection operation
      status = xnn_define_fully_connected(
        subgraph,
        /*output_min=*/-INFINITY,
        /*output_max=*/INFINITY,
        /*input_id=*/input_id,
        /*filter_id=*/w3_weight_id,
        /*bias_id=*/XNN_INVALID_VALUE_ID,  // No bias
        /*output_id=*/up_output_id,
        /*flags=*/0);
      if (status != xnn_status_success) {
        fprintf(stderr, "xnn_define_fully_connected failed: %d\n", status);
        return 1;
      }


    // SiLU activation on gate projection (implemented as sigmoid followed by multiply)
    // Define sigmoid output for SiLU
    uint32_t sigmoid_output_id;
    {
      std::vector<size_t> sigmoid_output_dims = {1, inter_dim};
      status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        /*num_dims=*/sigmoid_output_dims.size(),
        /*dims=*/sigmoid_output_dims.data(),
        /*data=*/nullptr,
        /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0,
        &sigmoid_output_id);
      if (status != xnn_status_success) {
        fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
        return 1;
      }
    }

    status = xnn_define_unary(
      subgraph,
      xnn_unary_sigmoid,
      /*params=*/nullptr,
      gate_output_id,
      sigmoid_output_id,
      /*flags=*/0);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_define_unary failed: %d\n", status);
      return 1;
    }
  

    // Define SiLU output
    uint32_t silu_output_id;
    {
      std::vector<size_t> silu_output_dims = {1, inter_dim};
      status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        /*num_dims=*/silu_output_dims.size(),
        /*dims=*/silu_output_dims.data(),
        /*data=*/nullptr,
        /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0,
        &silu_output_id);
      if (status != xnn_status_success) {
        fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
        return 1;
      }
    }

    // Define SiLU multiply operation: (w1 @ input) * sigmoid(w1 @ input)

    status = xnn_define_multiply2(
      subgraph,
      /*output_min=*/-INFINITY,
      /*output_max=*/INFINITY,
      /*input1_id=*/gate_output_id,
      /*input2_id=*/sigmoid_output_id,
      /*output_id=*/silu_output_id,
      /*flags=*/0);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_define_multiply2 failed: %d\n", status);
      return 1;
    }
  

    // Define gated intermediate output
    uint32_t gated_intermediate_output_id;
    {
      std::vector<size_t> gated_intermediate_output_dims = {1, inter_dim};
      status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        /*num_dims=*/gated_intermediate_output_dims.size(),
        /*dims=*/gated_intermediate_output_dims.data(),
        /*data=*/nullptr,
        /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0,
        &gated_intermediate_output_id);
      if (status != xnn_status_success) {
        fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
        return 1;
      }
    }

    // Define gated intermediate multiply operation: (SiLU(W1 @ input) * (W3 @ input))
    status = xnn_define_multiply2(
      subgraph,
      /*output_min=*/-INFINITY,
      /*output_max=*/INFINITY,
      /*input1_id=*/silu_output_id,
      /*input2_id=*/up_output_id,
      /*output_id=*/gated_intermediate_output_id,
      /*flags=*/0);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_define_multiply2 failed: %d\n", status);
      return 1;
    }


    // define down projection: w2 @ (SiLU(W1 @ input) * (W3 @ input))
    uint32_t w2_weight_id;
    {
      std::vector<size_t> w2_weight_dims = {output_dim, inter_dim};
      status = xnn_define_tensor_value(
        subgraph,
        xnn_datatype_fp32,
        /*num_dims=*/w2_weight_dims.size(),
        /*dims=*/w2_weight_dims.data(),
        /*data=*/w2_weight_data,
        /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0,
        &w2_weight_id);
      if (status != xnn_status_success) {
        fprintf(stderr, "xnn_define_tensor_value failed: %d\n", status);
        return 1;
      }
    }

    // Define down projection operation
    status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-INFINITY,
      /*output_max=*/INFINITY,
      /*input_id=*/gated_intermediate_output_id,
      /*filter_id=*/w2_weight_id,
      /*bias_id=*/XNN_INVALID_VALUE_ID,  // No bias
      /*output_id=*/output_id,
      /*flags=*/0);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_define_fully_connected failed: %d\n", status);
      return 1;
    }


    // Setting up a workspace and runtime for this SwiGLU operation

    // Create a workspace for this SwiGLU runtime
    xnn_workspace_t xnn_workspace;
    status = xnn_create_workspace(&xnn_workspace);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_create_workspace failed: %d\n", status);
      return 1;
    }

    xnn_runtime_t runtime;
    status = xnn_create_runtime_v4(
      subgraph,
      /*weights_cache=*/nullptr,
      /*workspace=*/xnn_workspace,
      /*threadpool=*/NULL,
      /*flags=*/0,
      &runtime);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_create_runtime_v4 failed: %d\n", status);
      return 1;
    }
  

    // Setting up external values and reshaping the runtime

    float input_data[BATCH_SIZE * INPUT_DIM] = { 1.0f, 2.0f, 3.0f };
    float output_data[BATCH_SIZE * OUTPUT_DIM];

    // Setup external tensors
    std::vector<xnn_external_value> external_values(2); // 2 external values: input and output

    // Input tensor (external ID 0)
    external_values[0].id = 0;
    external_values[0].data = input_data;

    // Output tensor (external ID 1)
    external_values[1].id = 1;
    external_values[1].data = output_data;


    // Reshaping 

    std::vector<size_t> input_dims = {BATCH_SIZE, INPUT_DIM};
    status = xnn_reshape_external_value(runtime, 0, input_dims.size(), input_dims.data());
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_reshape_external_value failed: %d\n", status);
      return 1;
    }
    std::vector<size_t> output_dims = {BATCH_SIZE, OUTPUT_DIM};
    status = xnn_reshape_external_value(runtime, 1, output_dims.size(), output_dims.data());
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_reshape_external_value failed: %d\n", status);
      return 1;
    }
    status = xnn_reshape_runtime(runtime);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_reshape_runtime failed: %d\n", status);
      return 1;
    }


    // Setup the runtime

    status = xnn_setup_runtime_v2(runtime, external_values.size(), external_values.data());
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_setup_runtime_v2 failed: %d\n", status);
      return 1;
    }
    status = xnn_invoke_runtime(runtime);
    if (status != xnn_status_success) {
      fprintf(stderr, "xnn_invoke_runtime failed: %d\n", status);
      return 1;
    }

  // 8. Inspect result
  printf("Output: [%f, %f]\n", output_data[0], output_data[1]);


  xnn_delete_runtime(runtime);
  xnn_release_workspace(xnn_workspace);
  xnn_delete_subgraph(subgraph);
  xnn_deinitialize();
  return 0;
}
