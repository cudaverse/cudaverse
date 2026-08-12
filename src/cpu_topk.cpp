#include <cmath>
#include <new>
#include <queue>
#include <vector>

#include <R.h>
#include <Rinternals.h>

namespace {

struct Candidate {
  double distance;
  int index;
};

struct BetterCandidate {
  bool operator()(const Candidate& left, const Candidate& right) const {
    if (left.distance < right.distance) return true;
    if (left.distance > right.distance) return false;
    return left.index < right.index;
  }
};

int scalar_integer(SEXP value, const char* name) {
  if (TYPEOF(value) != INTSXP || XLENGTH(value) != 1) {
    Rf_error("`%s` must be one integer.", name);
  }
  int result = INTEGER(value)[0];
  if (result == NA_INTEGER) {
    Rf_error("`%s` must be one integer.", name);
  }
  return result;
}

}  // namespace

extern "C" SEXP C_cudaverse_cpu_stable_topk(SEXP distance_sexp,
                                              SEXP rows_sexp,
                                              SEXP k_sexp) {
  if (!Rf_isReal(distance_sexp) || !Rf_isMatrix(distance_sexp)) {
    Rf_error("`distance` must be a double matrix.");
  }
  if (TYPEOF(rows_sexp) != INTSXP) {
    Rf_error("`rows` must be an integer vector.");
  }

  SEXP dimensions = Rf_getAttrib(distance_sexp, R_DimSymbol);
  int query_count = INTEGER(dimensions)[0];
  int candidate_count = INTEGER(dimensions)[1];
  int k = scalar_integer(k_sexp, "k");
  if (query_count < 1 || candidate_count < 2 || k < 1 ||
      k >= candidate_count || XLENGTH(rows_sexp) != query_count) {
    Rf_error("The stable top-k dimensions are invalid.");
  }

  const double* distance = REAL(distance_sexp);
  const int* rows = INTEGER(rows_sexp);
  R_xlen_t value_count = XLENGTH(distance_sexp);
  for (R_xlen_t index = 0; index < value_count; ++index) {
    if (ISNAN(distance[index])) {
      Rf_error("`distance` must not contain NA or NaN values.");
    }
  }
  for (int query = 0; query < query_count; ++query) {
    if (rows[query] < 1 || rows[query] > candidate_count) {
      Rf_error("`rows` contains an index outside the candidate matrix.");
    }
  }

  SEXP index_result = PROTECT(Rf_allocMatrix(INTSXP, query_count, k));
  SEXP distance_result = PROTECT(Rf_allocMatrix(REALSXP, query_count, k));
  int* output_index = INTEGER(index_result);
  double* output_distance = REAL(distance_result);

  for (int query = 0; query < query_count; ++query) {
    bool allocation_failed = false;
    try {
      {
        // BetterCandidate makes priority_queue::top() the worst retained
        // value, so a better (distance, original index) pair replaces it in
        // O(log k).
        std::priority_queue<Candidate, std::vector<Candidate>, BetterCandidate>
            selected;
        int self = rows[query];
        for (int candidate = 1; candidate <= candidate_count; ++candidate) {
          if (candidate == self) continue;
          Candidate value{
              distance[query + query_count * (candidate - 1)], candidate};
          if (static_cast<int>(selected.size()) < k) {
            selected.push(value);
          } else if (BetterCandidate{}(value, selected.top())) {
            selected.pop();
            selected.push(value);
          }
        }
        for (int position = k - 1; position >= 0; --position) {
          const Candidate value = selected.top();
          selected.pop();
          output_index[query + query_count * position] = value.index;
          output_distance[query + query_count * position] = value.distance;
        }
      }
    } catch (const std::bad_alloc&) {
      allocation_failed = true;
    }
    if (allocation_failed) Rf_error("CPU stable top-k allocation failed.");
    R_CheckUserInterrupt();
  }

  SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_VECTOR_ELT(result, 0, index_result);
  SET_VECTOR_ELT(result, 1, distance_result);
  SET_STRING_ELT(names, 0, Rf_mkChar("index"));
  SET_STRING_ELT(names, 1, Rf_mkChar("distance"));
  Rf_setAttrib(result, R_NamesSymbol, names);

  UNPROTECT(4);
  return result;
}
