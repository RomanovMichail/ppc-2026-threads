#include "romanov_m_matrix_ccs/all/include/ops_all.hpp"

#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "romanov_m_matrix_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_m_matrix_ccs {

RomanovMMatrixCCSALL::RomanovMMatrixCCSALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RomanovMMatrixCCSALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int res = 0;
  if (rank == 0) {
    auto &left = GetInput().first;
    auto &right = GetInput().second;
    res = (left.cols_num == right.rows_num && left.cols_num > 0) ? 1 : 0;
  }
  MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return res == 1;
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
      size_t row_i = a.row_inds[ka];
      if (!row_mask[row_i]) {
        row_mask[row_i] = true;
        active_rows.push_back(row_i);
      }
      accumulator[row_i] += a.vals[ka] * v_b;
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

void RomanovMMatrixCCSALL::SyncMatrixData(int rank, MatrixCCS &a, MatrixCCS &b) {
  std::vector<int> dims(3, 0);
  if (rank == 0) {
    dims[0] = static_cast<int>(a.rows_num);
    dims[1] = static_cast<int>(a.cols_num);
    dims[2] = static_cast<int>(b.cols_num);
  }
  MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    a.rows_num = static_cast<size_t>(dims[0]);
    a.cols_num = static_cast<size_t>(dims[1]);
    b.rows_num = static_cast<size_t>(dims[1]);
    b.cols_num = static_cast<size_t>(dims[2]);
    a.col_ptrs.resize(a.cols_num + 1);
    b.col_ptrs.resize(b.cols_num + 1);
  }

  MPI_Bcast(a.col_ptrs.data(), static_cast<int>(a.cols_num + 1), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    a.row_inds.resize(a.col_ptrs[a.cols_num]);
    a.vals.resize(a.col_ptrs[a.cols_num]);
  }
  MPI_Bcast(a.row_inds.data(), static_cast<int>(a.col_ptrs[a.cols_num]), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(a.vals.data(), static_cast<int>(a.col_ptrs[a.cols_num]), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(b.col_ptrs.data(), static_cast<int>(b.cols_num + 1), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    b.row_inds.resize(b.col_ptrs[b.cols_num]);
    b.vals.resize(b.col_ptrs[b.cols_num]);
  }
  MPI_Bcast(b.row_inds.data(), static_cast<int>(b.col_ptrs[b.cols_num]), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(b.vals.data(), static_cast<int>(b.col_ptrs[b.cols_num]), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool RomanovMMatrixCCSALL::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MatrixCCS a_mat = GetInput().first;
  MatrixCCS b_mat = GetInput().second;
  SyncMatrixData(rank, a_mat, b_mat);

  int total_cols = static_cast<int>(b_mat.cols_num);
  int chunk = total_cols / size;
  int remainder = total_cols % size;
  int start_col = (rank * chunk) + std::min(rank, remainder);
  int end_col = start_col + chunk + ((rank < remainder) ? 1 : 0);
  int local_count = end_col - start_col;

  std::vector<std::vector<double>> local_v(local_count);
  std::vector<std::vector<size_t>> local_r(local_count);

  tbb::parallel_for(0, local_count, [&](int i) {
    MultiplyColumn(static_cast<size_t>(start_col + i), a_mat, b_mat, local_v[i], local_r[i]);
  });

  CollectResults(rank, size, chunk, remainder, local_v, local_r);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

void RomanovMMatrixCCSALL::CollectResults(int rank, int size, int chunk, int remainder,
                                          std::vector<std::vector<double>> &local_v,
                                          std::vector<std::vector<size_t>> &local_r) {
  auto &c_mat = GetOutput();
  int total_cols = (rank == 0) ? static_cast<int>(GetInput().second.cols_num) : 0;

  if (rank == 0) {
    std::vector<std::vector<double>> all_v(total_cols);
    std::vector<std::vector<size_t>> all_r(total_cols);
    int start = std::min(rank, remainder) + rank * chunk;
    for (size_t i = 0; i < local_v.size(); ++i) {
      all_v[start + i] = std::move(local_v[i]);
      all_r[start + i] = std::move(local_r[i]);
    }
    for (int p = 1; p < size; ++p) {
      int p_start = (p * chunk) + std::min(p, remainder);
      int p_cols = (chunk + ((p < remainder) ? 1 : 0));
      for (int i = 0; i < p_cols; ++i) {
        int nnz = 0;
        MPI_Recv(&nnz, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        all_v[p_start + i].resize(nnz);
        all_r[p_start + i].resize(nnz);
        if (nnz > 0) {
          MPI_Recv(all_v[p_start + i].data(), nnz, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(all_r[p_start + i].data(), nnz, MPI_UNSIGNED_LONG, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }
    c_mat.rows_num = GetInput().first.rows_num;
    c_mat.cols_num = total_cols;
    c_mat.col_ptrs.assign(total_cols + 1, 0);
    for (int j = 0; j < total_cols; ++j) {
      c_mat.col_ptrs[j + 1] = c_mat.col_ptrs[j] + all_v[j].size();
      c_mat.vals.insert(c_mat.vals.end(), all_v[j].begin(), all_v[j].end());
      c_mat.row_inds.insert(c_mat.row_inds.end(), all_r[j].begin(), all_r[j].end());
    }
    c_mat.nnz = c_mat.vals.size();
  } else {
    for (size_t i = 0; i < local_v.size(); ++i) {
      int nnz = static_cast<int>(local_v[i].size());
      MPI_Send(&nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      if (nnz > 0) {
        MPI_Send(local_v[i].data(), nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(local_r[i].data(), nnz, MPI_UNSIGNED_LONG, 0, 2, MPI_COMM_WORLD);
      }
    }
  }
}

bool RomanovMMatrixCCSALL::PostProcessingImpl() {
  return true;
}

}  // namespace romanov_m_matrix_ccs
