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

// Normalizing adjacency matrix
void normalize_adj(float *adj, float *norm_adj, int num_nodes) {
    float *degree = (float *)malloc(num_nodes * sizeof(float));
    
    for (int i = 0; i < num_nodes; ++i) {
        degree[i] = 0;
        for (int j = 0; j < num_nodes; ++j)
            degree[i] += adj[i * num_nodes + j];
    }
    
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (degree[i] > 0 && degree[j] > 0)
                norm_adj[i * num_nodes + j] = adj[i * num_nodes + j] / sqrt(degree[i] * degree[j]);
            else
                norm_adj[i * num_nodes + j] = 0;
        }
    }
    
    free(degree);
}

// Graph Transformer layer forward pass
void graph_transformer_layer(float *x, float *adj, float *w_q, float *w_k, float *w_v, float *w_out, float *out, int num_nodes, int input_dim, int output_dim) {


    float *q = (float *)malloc(num_nodes * output_dim * sizeof(float));
    float *k = (float *)malloc(num_nodes * output_dim * sizeof(float));
    float *v = (float *)malloc(num_nodes * output_dim * sizeof(float));
    float *attn_scores = (float *)malloc(num_nodes * num_nodes * sizeof(float));
    float *attn_sum = (float *)malloc(num_nodes * sizeof(float));
    float *z = (float *)malloc(num_nodes * output_dim * sizeof(float));
    float *norm_adj = (float *)malloc(num_nodes * num_nodes * sizeof(float));
    
    // Normalize adjacency matrix
    normalize_adj(adj, norm_adj, num_nodes);
    
    // Q, K, V
    matmul(x, w_q, q, num_nodes, input_dim, output_dim);
    matmul(x, w_k, k, num_nodes, input_dim, output_dim);
    matmul(x, w_v, v, num_nodes, input_dim, output_dim);
    
    // Attention mechanism
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (norm_adj[i * num_nodes + j] > 0) {
                attn_scores[i * num_nodes + j] = exp(q[i] * k[j]);
                attn_sum[i] += attn_scores[i * num_nodes + j];
            } else {
                attn_scores[i * num_nodes + j] = 0;
            }
        }
    }
    
    // Aggregate node features
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            if (norm_adj[i * num_nodes + j] > 0) {
                for (int k = 0; k < output_dim; ++k) {
                    z[i * output_dim + k] += (attn_scores[i * num_nodes + j] / attn_sum[i]) * v[j * output_dim + k];
                }
            }
        }
    }
    
    // Output layer
    matmul(z, w_out, out, num_nodes, output_dim, output_dim);
    
    free(q);
    free(k);
    free(v);
    free(attn_scores);
    free(attn_sum);
    free(z);
    free(norm_adj);
}


int main() {
    // Example usage of the graph_transformer_layer function
    int num_nodes = 3;
    int input_dim = 2;
    int output_dim = 2;

    float x[6] = {1, 2, 3, 4, 5, 6};  // Example input features
    float adj[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};  // Example adjacency matrix
    float w_q[4] = {1, 0, 0, 1};  // Example weight matrix for Q
    float w_k[4] = {1, 0, 0, 1};  // Example weight matrix for K
    float w_v[4] = {1, 0, 0, 1};  // Example weight matrix for V
    float w_out[4] = {1, 0, 0, 1};  // Example weight matrix for output
    float out[6];  // Output array

    graph_transformer_layer(x, adj, w_q, w_k, w_v, w_out, out, num_nodes, input_dim, output_dim);

    // Print the output
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            printf("%f ", out[i * output_dim + j]);
        }
        printf("\n");
    }

    return 0;
}