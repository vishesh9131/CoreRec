#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to perform matrix multiplication
void matmul(float *a, float *b, float *c, int m, int n, int p) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j) {
            c[i * p + j] = 0;
            for (int k = 0; k < n; ++k)
                c[i * p + j] += a[i * n + k] * b[k * p + j];
        }
}

// LeakyReLU activation function
float leaky_relu(float x, float alpha) {
    return x > 0 ? x : alpha * x;
}

// GAT layer forward pass
void gat_layer(float *x, float *adj, float *a_src, float *a_dst, float *w, float *out, int num_nodes, int input_dim, int output_dim, float alpha) {
    float *h = (float *)malloc(num_nodes * output_dim * sizeof(float));
    float *attn_scores = (float *)malloc(num_nodes * num_nodes * sizeof(float));
    float *attn_sum = (float *)malloc(num_nodes * sizeof(float));
    
    // XW
    matmul(x, w, h, num_nodes, input_dim, output_dim);
    
    // Attention mechanism
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (adj[i * num_nodes + j] > 0) {
                float attn = leaky_relu(a_src[i] + a_dst[j], alpha);
                attn_scores[i * num_nodes + j] = exp(attn);
                attn_sum[i] += attn_scores[i * num_nodes + j];
            } else {
                attn_scores[i * num_nodes + j] = 0;
            }
        }
    }
    
    // Aggregate node features
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (adj[i * num_nodes + j] > 0) {
                for (int k = 0; k < output_dim; ++k) {
                    out[i * output_dim + k] += (attn_scores[i * num_nodes + j] / attn_sum[i]) * h[j * output_dim + k];
                }
            }
        }
    }
    
    free(h);
    free(attn_scores);
    free(attn_sum);
}

int main() {
    // Example usage of the gat_layer function
    int num_nodes = 3;
    int input_dim = 2;
    int output_dim = 2;
    float alpha = 0.2;

    float x[6] = {1, 2, 3, 4, 5, 6};  // Example input features
    float adj[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};  // Example adjacency matrix
    float a_src[3] = {0.1, 0.2, 0.3};  // Example source attention weights
    float a_dst[3] = {0.1, 0.2, 0.3};  // Example destination attention weights
    float w[4] = {1, 0, 0, 1};  // Example weight matrix
    float out[6] = {0};  // Output array

    gat_layer(x, adj, a_src, a_dst, w, out, num_nodes, input_dim, output_dim, alpha);

    // Print the output
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            printf("%f ", out[i * output_dim + j]);
        }
        printf("\n");
    }

    return 0;
}