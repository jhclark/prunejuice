#include <vector>
#include <cassert>
#include <functional>
#include <gtest/gtest.h>

#include "fast_oscar.h"

using namespace std;

double gold_function(double x) {
  return 10.0 * x * x + 100.0 * x + 1000.0;
}

// return the features for a given input
// a perfect optimizer should be able to exactly fit gold_func using this feature set:
// w_0 = 0
// w_1 = 1000
// w_2 = 10
// w_3 = 100
// (they're out of order to check for sorting-related bugs in the OSCAR proximal step)
void perfect_feature_function(const double x, vector<double>* H) {
  H->resize(4);
  
  (*H)[0] = 0.0;
  (*H)[1] = 1.0;
  (*H)[2] = x * x;
  (*H)[3] = x;
}

// return the features for a given input
void approx_feature_function(const double x, vector<double>* H) {
  H->resize(4);
  
  (*H)[0] = 1;

  if (x > 0)
    (*H)[1] = 1000;
  else
    (*H)[1] = 0;

  (*H)[2] = x;
  
  if (x > 4)
    (*H)[3] = 1500;
  else
    (*H)[3] = 0;

  (*H)[3] = x * x;
}

double model_function(const vector<double>& W, const vector<double>& H) {
  assert(W.size() == H.size());
  double result = 0.0;
  for (size_t i = 0; i < W.size(); i++) {
    result += W.at(i) * H.at(i);
  }
  return result;
}

// returns the gradient if G if non-null
double loss_function(
    const vector<double>& W,
    const vector<double>& X,
    const vector<double>& Y,
    function<void (double, vector<double>*)> feature_func,
    vector<double>* G) {
  assert(X.size() == Y.size());

  double loss = 0.0;
  for (size_t i = 0; i < X.size(); i++) {
    double x = X.at(i);
    double y_gold = Y.at(i);

    vector<double> H;
    feature_func(x, &H);
    double y_predicted = model_function(W, H);
    double diff = y_gold - y_predicted;
    
    if (G != NULL) {
      assert(G->size() == W.size());
      for (size_t j = 0; j < W.size(); j++) {
        (*G)[j] += 2 * H.at(j) * diff;
	//cerr << "g_" << j << " = " << G->at(j) << endl;
      }
    }

    cerr << "Example: " << y_gold << " Predicted: " << y_predicted << " Diff: " << diff << endl;
    loss += diff * diff;
  }

  // we normalize the gradient due to its large magnitudes
  // this is primarily because this toy example is a regression problem with a
  // large dynamic range of features, gradients, and outputs, which won't be such a big issue in log-loss classification
  const bool NORMALIZE_GRADIENT = true;
  if (NORMALIZE_GRADIENT) {
    double sum = abs(std::accumulate(G->cbegin(), G->cend(), 0.0, std::plus<double>()));
    cerr << "Normalizing gradient..." << endl;
    for (size_t j = 0; j < W.size(); j++) {
      (*G)[j] /= sum;
      cerr << "g_" << j << " = " << G->at(j) << endl;
    }
  }

  cerr << "Loss is " << loss << endl;
  return loss;
}

void get_data(vector<double>* X, vector<double>* Y) {
  X->clear();
  X->push_back(0); // 1000
  X->push_back(1); // 1110
  X->push_back(5); // 1750
  X->push_back(10); // 3000

  for (double x : *X) {
    Y->push_back(gold_function(x));
  }
}

TEST(OptimizeTest, OptimizeTest) {
  vector<double> X;
  vector<double> Y;
  get_data(&X, &Y);

  double NUM_FEATURES = 4;
  int NUM_ITERATIONS = 100 * 1000;
  double BUFFER_SIZE = -1;

  double C1 = 0.0; // L1 regularizer
  double C_inf = 0.0; // pairwise L_infinity regularizer
  double init_learning_rate = 1.0;
  // gradient magnitutes are huge for this regression problem due to large y's and squared loss
  // this will generally not be such a big problem for log loss classifiers
  double nonadapted_learning_rate = 1.0;
  vector<double> init_weights;
  init_weights.resize(NUM_FEATURES);
  vector<double> learned_weights;
  learned_weights.resize(NUM_FEATURES);

  OscarAdaGradOptimize(C1, C_inf, init_learning_rate, nonadapted_learning_rate, BUFFER_SIZE, NUM_ITERATIONS, init_weights, &learned_weights,
    [&](const vector<double>& W, vector<double>* G) {
       loss_function(W, X, Y, perfect_feature_function, G);
    });

  for (size_t i = 0; i < learned_weights.size(); i++) {
    cerr << "w_" << i << " = " << learned_weights.at(i) << endl;
  }

  //EXPECT_EQ(2.0, sqrt(4.0));
}

// TODO: Design a function with a complex set of parameters to optimize
//       along with a clear multi-parameter solution (approximation?)
// check for recovering AdaGrad with all regularization set to zero
// check for having fewer non-zero parameters with non-zero L1 constant
// check for having fewer unique parameters with L_inf constant
