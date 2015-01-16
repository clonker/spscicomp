#ifndef __SPSCICOMP_HiddenMarkovModel_H__
#define __SPSCICOMP_HiddenMarkovModel_H__

#include <istream>
#include <vector>
#include <exception>
#include <boost/multi_array.hpp>

namespace spscicomp 
{

class bad_file_syntax : public std::runtime_error {
public:
  bad_file_syntax(const std::string &what_arg)
  : std::runtime_error(what_arg) { }
  virtual ~bad_file_syntax() throw() { }
};

template <typename T>
struct hidden_markov_model 
{
  typedef boost::multi_array<T, 2> matrix;
  typedef std::vector<T> vector;

  matrix A;
  matrix B;
  vector pi;
  size_t N;
  size_t M;

  friend std::istream &operator>>(std::istream &in, hidden_markov_model<T> &hmm)
  {
    size_t N, M;
    if (in >> N && in >> M) {
      hmm.N = N;
      hmm.M = M;
      matrix::extent_gen extents;
      hmm.A.resize(extents[N][N]);
      hmm.B.resize(extents[N][M]);
      hmm.pi.resize(N);

      for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
          if (!(in >> hmm.A[i][j]))
            throw bad_file_syntax("Could not read transition matrix.");

      for (size_t i = 0; i < N; i++)
        for (size_t k = 0; k < M; k++)
          if (!(in >> hmm.B[i][k]))
            throw bad_file_syntax("Could not read symbol probabilities");

      for (size_t i = 0; i < N; i++)
        if (!(in >> hmm.pi[i]))
          throw bad_file_syntax("Could not read initial distribution.");
    } else
      throw bad_file_syntax("Could not read dimension N or M.");
    return in;
  }

  friend std::ostream &operator<<(std::ostream &out, const hidden_markov_model<T> &hmm)
  {
    out << hmm.N << " " << hmm.M << std::endl;
    for (size_t i = 0; i < hmm.N; i++) {
      for (size_t j = 0; j < hmm.N; j++)
        out << hmm.A[i][j] << " ";
      out << std::endl;
    }
    for (size_t i = 0; i < hmm.N; i++) {
      for (size_t j = 0; j < hmm.M; j++)
        out << hmm.B[i][j] << " ";
      out << std::endl;
    }
    for (size_t i = 0; i < hmm.N; i++)
      out << hmm.pi[i] << " ";
    return out;
  }
};

} // namespace spscicomp

#endif // __SPSCICOMP_HiddenMarkovModel_H__