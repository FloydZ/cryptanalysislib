#include "cuda/cuda.h"
#include "cuda/gaus.cu"

#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

//#include "list.h"
//
//using DecodingValue = Value_T<BinaryContainer<G_k + G_l>>;
//using DecodingLabel = Label_T<BinaryContainer<G_n - G_k>>;
//using DecodingMatrix = mzd_t *;
//using DecodingElement = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;

size_t _m4ri_rows_to_32_bit(mzd_t* A, uint32_t* rows) {
	size_t size_array = A->nrows * A->width * 2 * sizeof(uint32_t);

	for(int x = 0; x < A->nrows; x++) {
		for (int y = 0; y < A->width; y++) {
			rows[(x * A->width + y) * 2 + 1] = (uint32_t) ((A->rows[x][y] & 0xFFFFFFFF00000000LL) >> 32);
			rows[(x * A->width + y) * 2] = (uint32_t) (A->rows[x][y] & 0xFFFFFFFFLL);
		}
	}

	return size_array;
}

void _32_bit_to_m4ri_rows(mzd_t* A, const uint32_t *rows) {
	for(int x = 0; x < A->nrows; x++) {
		for (int y = 0; y < A->width; y++) {
			A->rows[x][y] = (((word) rows[(x * A->width + y) * 2 + 1]) << 32) | rows[(x * A->width + y) * 2];
		}
	}
}

int main(int argc, char **argv) {
	for(int k = 0; k < 10; k++) {
		int n = 64;
		mzd_t* L = mzd_init(n, n);
		mzd_set_ui(L, 1);

		for(int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				bool b = rand() % 2;
				if (i != j && b)
					mzd_row_add(L, i, j);
			}
		}
		uint32_t* rows = nullptr;
		//create new Matrix L_extended

		//const rci_t n  = L->nrows;
		const rci_t nr = m4ri_radix * L->width;
		mzd_t *L_new = mzd_init(n, 2 * nr);

		mzd_t *LW = mzd_init_window(L_new, 0, 0, n, n);
		mzd_t *L_inv = mzd_init_window(L_new, 0, nr, n, nr + n);

		mzd_copy(LW, L);
		mzd_set_ui(L_inv, 1);

		uint32_t* host_rows = nullptr;
		host_rows = (uint32_t*)calloc((L_new->nrows * L_new->width * 2), sizeof(uint32_t));

		size_t size_array = _m4ri_rows_to_32_bit(L_new, host_rows);

		checkCudaErrors(cudaMalloc(&rows, size_array));
		checkCudaErrors(cudaMemcpy(rows, host_rows, size_array, cudaMemcpyHostToDevice));

		uint32_t* to_xor;
		uint32_t* I_array;
		checkCudaErrors(cudaMalloc(&to_xor, 4 * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&I_array, 2 * sizeof(uint32_t)));


		//int permuted_until = integrated_gauss_jordan_cuda(rows, L_new->width * 2, L_new->nrows);
		gauss_jordan_cuda_integrated<<<1,dim3(L_new->nrows / 32 ,L_new->width * 2)>>>(rows,
																						to_xor,
																						I_array,
																						L_new->nrows,
																						0,
																						2,
																						L_new->width * 2,
																						n);


		checkCudaErrors(cudaMemcpy(host_rows, rows, size_array, cudaMemcpyDeviceToHost));


		cudaFree(rows);
		cudaFree(to_xor);
		cudaFree(I_array);

		_32_bit_to_m4ri_rows(L_new, host_rows);


		//if(permuted_until != -1)
		//    return -1;

		//__M4RI_DD_MZD(L_inv);

		mzd_t* L_inv_test = mzd_inv_m4ri(NULL, L, 0);

		// for(int i = 0; i < 64; i++)
		// {
		//     for(int j = 0; j < 64; j++)
		//     {
		//         printf("%d",mzd_read_bit(L, i, j));
		//     }
		//     printf("\t");
		//     for(int j = 0; j < 64; j++)
		//     {
		//         printf("%d",mzd_read_bit(L_new, i, j));
		//     }
		//     printf("\n");
		// }

		for(int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				if (mzd_read_bit(L_inv_test, i, j) != mzd_read_bit(L_inv, i, j)) {
					return -1;
				}
			}
		}
		mzd_free_window(LW);
		mzd_free_window(L_inv);
		mzd_free(L);
		mzd_free(L_inv_test);
	}

	for(int k = 0; k < 10; k++) {
		int n = 64;
		mzd_t* L = mzd_init(n, n);
		mzd_set_ui(L, 1);

		int r_row = rand() % n;

		mzd_row_add(L, r_row, r_row);

		for(int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				bool b = rand() % 2;
				if (i != j && b)
					mzd_row_add(L, i, j);
			}
		}

		uint32_t* rows = nullptr;
		//create new Matrix L_extended

		//const rci_t n  = L->nrows;
		const rci_t nr = m4ri_radix * L->width;
		mzd_t *L_new = mzd_init(n, 2 * nr);

		mzd_t *LW = mzd_init_window(L_new, 0, 0, n, n);
		mzd_t *L_inv = mzd_init_window(L_new, 0, nr, n, nr + n);

		mzd_copy(LW, L);
		mzd_set_ui(L_inv, 1);

		uint32_t* host_rows = nullptr;
		host_rows = (uint32_t*)calloc((L_new->nrows * L_new->width * 2), sizeof(uint32_t));

		size_t size_array = _m4ri_rows_to_32_bit(L_new, host_rows);

		cudaMalloc(&rows, size_array);
		cudaMemcpy(rows, host_rows, size_array, cudaMemcpyHostToDevice);

		//mzd_t* L_inv = gauss_jordan_cuda(L, n);
		//int permuted_until = integrated_gauss_jordan_cuda(rows, L_new->width * 2, L_new->nrows);
		uint32_t* empty_permutation;
		cudaMalloc(&empty_permutation, 1 * sizeof(uint32_t));

		int permuted_until = integrated_gauss_jordan_cuda(rows, L_new->width * 2, L_new->nrows);
		//gauss_jordan_cuda_integrated<<<dim3(1),dim3(16,16), 2*size_array>>>(rows, L_new->width * 2, L_new->nrows, 1, n, 0, empty_permutation, 0);

		//uint32_t permuted_until;
		//cudaMemcpy(&permuted_until, empty_permutation, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_rows, rows, size_array, cudaMemcpyDeviceToHost);
		cudaFree(rows);
		cudaFree(empty_permutation);

		_32_bit_to_m4ri_rows(L_new, host_rows);

		if(permuted_until > L_new->nrows)
		{
			return -1;
		}

		mzd_free_window(LW);
		mzd_free_window(L_inv);
		mzd_free(L);
	}
}