#include "romanov_m_matrix_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace romanov_m_matrix_ccs {

RomanovMMatrixCCSALL::RomanovMMatrixCCSALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RomanovMMatrixCCSALL::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto &left = GetInput().first;
    auto &right = GetInput().second;
    int res = (left.cols_num == right.rows_num && left.cols_num > 0) ? 1 : 0;
    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return res == 1;
  } else {
    int res;
    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return res == 1;
  }
}

bool RomanovMMatrixCCSALL::PreProcessingImpl() {
  return true;
}

void RomanovMMatrixCCSALL::MultiplyColumn(size_t col_index, const MatrixCCS &a, const MatrixCCS &b,
                                          std::vector<double> &temp_v, std::vector<size_t> &temp_r) {
  std::vector<double> accumulator(a.rows_num, 0.0);
  std::vector<bool> row_mask(a.rows_num, false);
  std::vector<size_t> active_rows;
  for (size_t kb = b.col_ptrs[col_index]; kb < b.col_ptrs[col_index + 1]; ++kb) {
    size_t k = b.row_inds[kb];
    double v_b = b.vals[kb];
    for (size_t ka = a.col_ptrs[k]; ka < a.col_ptrs[k + 1]; ++ka) {
      size_t i = a.row_inds[ka];
      if (!row_mask[i]) {
        row_mask[i] = true;
        active_rows.push_back(i);
      }
      accumulator[i] += a.vals[ka] * v_b;
    }
  }
  std::ranges::sort(active_rows);
  for (size_t row_idx : active_rows) {
    if (std::abs(accumulator[row_idx]) > 1e-12) {
      temp_v.push_back(accumulator[row_idx]);
      temp_r.push_back(row_idx);
    }
  }
}

bool RomanovMMatrixCCSALL::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto &c = GetOutput();
  if (rank == 0) {
    c.vals.clear();
    c.row_inds.clear();
  }
  MatrixCCS a = GetInput().first;
  MatrixCCS b = GetInput().second;
  int dims[3];
  if (rank == 0) {
    dims[0] = static_cast<int>(a.rows_num);
    dims[1] = static_cast<int>(a.cols_num);
    dims[2] = static_cast<int>(b.cols_num);
  }
  MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    a.rows_num = dims[0];
    a.cols_num = dims[1];
    b.rows_num = dims[1];
    b.cols_num = dims[2];
    a.col_ptrs.resize(a.cols_num + 1);
    b.col_ptrs.resize(b.cols_num + 1);
  }
  MPI_Bcast(a.col_ptrs.data(), a.cols_num + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    a.row_inds.resize(a.col_ptrs[a.cols_num]);
    a.vals.resize(a.col_ptrs[a.cols_num]);
  }
  MPI_Bcast(a.row_inds.data(), a.col_ptrs[a.cols_num], MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(a.vals.data(), a.col_ptrs[a.cols_num], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b.col_ptrs.data(), b.cols_num + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    b.row_inds.resize(b.col_ptrs[b.cols_num]);
    b.vals.resize(b.col_ptrs[b.cols_num]);
  }
  MPI_Bcast(b.row_inds.data(), b.col_ptrs[b.cols_num], MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(b.vals.data(), b.col_ptrs[b.cols_num], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int total_cols = static_cast<int>(b.cols_num);
  int chunk = total_cols / size;
  int remainder = total_cols % size;
  int start_col = rank * chunk + std::min(rank, remainder);
  int end_col = start_col + chunk + (rank < remainder ? 1 : 0);
  int local_cols_count = end_col - start_col;
  std::vector<std::vector<double>> local_temp_vals(local_cols_count);
  std::vector<std::vector<size_t>> local_temp_rows(local_cols_count);
  tbb::parallel_for(0, local_cols_count,
                    [&](int i) { MultiplyColumn(start_col + i, a, b, local_temp_vals[i], local_temp_rows[i]); });
  if (rank == 0) {
    c.rows_num = a.rows_num;
    c.cols_num = b.cols_num;
    c.col_ptrs.assign(c.cols_num + 1, 0);
    std::vector<std::vector<double>> all_vals(b.cols_num);
    std::vector<std::vector<size_t>> all_rows(b.cols_num);
    for (int i = 0; i < local_cols_count; ++i) {
      all_vals[start_col + i] = std::move(local_temp_vals[i]);
      all_rows[start_col + i] = std::move(local_temp_rows[i]);
    }
    for (int p = 1; p < size; ++p) {
      int p_start = p * chunk + std::min(p, remainder);
      int p_end = p_start + chunk + (p < remainder ? 1 : 0);
      int p_cols = p_end - p_start;
      for (int i = 0; i < p_cols; ++i) {
        int nnz_p;
        MPI_Recv(&nnz_p, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        all_vals[p_start + i].resize(nnz_p);
        all_rows[p_start + i].resize(nnz_p);
        if (nnz_p > 0) {
          MPI_Recv(all_vals[p_start + i].data(), nnz_p, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(all_rows[p_start + i].data(), nnz_p, MPI_UINT64_T, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }
    size_t total_nnz = 0;
    for (size_t j = 0; j < b.cols_num; ++j) {
      c.col_ptrs[j] = total_nnz;
      total_nnz += all_vals[j].size();
      c.vals.insert(c.vals.end(), all_vals[j].begin(), all_vals[j].end());
      c.row_inds.insert(c.row_inds.end(), all_rows[j].begin(), all_rows[j].end());
    }
    c.col_ptrs[b.cols_num] = total_nnz;
    c.nnz = total_nnz;
  } else {
    for (int i = 0; i < local_cols_count; ++i) {
      int nnz_local = static_cast<int>(local_temp_vals[i].size());
      MPI_Send(&nnz_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      if (nnz_local > 0) {
        MPI_Send(local_temp_vals[i].data(), nnz_local, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(local_temp_rows[i].data(), nnz_local, MPI_UINT64_T, 0, 2, MPI_COMM_WORLD);
      }
    }
  }
  int final_res[3];
  if (rank == 0) {
    final_res[0] = static_cast<int>(c.rows_num);
    final_res[1] = static_cast<int>(c.cols_num);
    final_res[2] = static_cast<int>(c.nnz);
  }
  MPI_Bcast(final_res, 3, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    c.rows_num = final_res[0];
    c.cols_num = final_res[1];
    c.nnz = final_res[2];
    c.col_ptrs.resize(c.cols_num + 1);
    c.vals.resize(c.nnz);
    c.row_inds.resize(c.nnz);
  }
  MPI_Bcast(c.col_ptrs.data(), c.cols_num + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (c.nnz > 0) {
    MPI_Bcast(c.vals.data(), c.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(c.row_inds.data(), c.nnz, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool RomanovMMatrixCCSALL::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_m_matrix_ccs
