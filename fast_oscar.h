#include <vector>
#include <stack>
#include <queue>
#include <functional>
#include <limits>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace std;

//namespace fast_oscar {
// class AdaGradOscarOptimizer {

struct OscarGroup {
  vector<size_t> _orig_indices;
  vector<size_t> _sorted_indices;
  double _init_value;

  size_t Size() const {
    assert(_orig_indices.size() == _sorted_indices.size());
    return _orig_indices.size();
  }

  void MergeWith(OscarGroup& that) {
    assert(_orig_indices.size() == _sorted_indices.size());
    _orig_indices.insert(_orig_indices.begin(), that._orig_indices.begin(), that._orig_indices.end());
    _sorted_indices.insert(_sorted_indices.begin(), that._sorted_indices.begin(), that._sorted_indices.end());
  }
};

// g is G[i], sum of historical squared gradients
double EffectiveLearningRate(const double init_learning_rate, const double nonadapted_learning_rate, const double sum_g2) {
    double sqrt_G = sqrt(sum_g2);
    if (sqrt_G > 0.0)
      return nonadapted_learning_rate + init_learning_rate / sqrt_G;
    else
      return nonadapted_learning_rate;
}

// prepare inputs for OSCAR proximal step, according to the diagonal AdaGrad update strategy
void OscarPrepareProximalInputs(const vector<double>& pre_proximal_weights, vector<double>* a) {
  assert(a != NULL);
  assert(pre_proximal_weights.size() == a->size());
  const size_t N = pre_proximal_weights.size();
  for (size_t i = 0; i < N; i++) {
    (*a)[i] = abs(pre_proximal_weights.at(i));
  }
}

// zero-safe implementation of C99's copysign
// (floats are signed, which can lead to confusing results)
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

void OscarWeightsFromProximalSolution(const vector<double>& z, const vector<double>& pre_proximal_weights, vector<double>* new_weights) {
  assert (new_weights != NULL);
  assert (z.size() == pre_proximal_weights.size());
  assert (z.size() == new_weights->size());

  size_t N = z.size();
  for (size_t i = 0; i < N; i++) {
    int polarity = sgn(pre_proximal_weights.at(i));
    cerr << "Polarity_" << i << " " << polarity << "; " << pre_proximal_weights.at(i) << endl;
    (*new_weights)[i] = polarity * z.at(i);
  }
}

// w: Measures contribution of regularizers
// used for determining an actual or hypothetical common parameter value within a group
double ComputeW(
    const double C1, // constant for L1 regularizer
    const double C_inf, // constant for pairwise L_inf regularizer
    const size_t N, // dimensionality / feature count
    const size_t i // the index of interest **in the sorted list** of a's
  ) {
  return C1 + C_inf * (N-i);
}

// compute the common value for a group or proposed group
// equation (15)
double ComputeCommonValue(
    const vector<double>& a,
    const double C1, // constant for L1 regularizer
    const double C_inf, // constant for pairwise L_inf regularizer
    const size_t N, // dimensionality / feature count
    const OscarGroup& group) {

  double numerator = 0.0;
  for (size_t k = 0; k < group._sorted_indices.size(); k++) {
    size_t orig_idx = group._orig_indices.at(k); // for accessing a's, etc.
    size_t sorted_idx = group._sorted_indices.at(k); // for determining impact under pairwise L_inf regularizer
    numerator += a.at(orig_idx) - ComputeW(C1, C_inf, N, sorted_idx);
  }
  double v =  numerator / group.Size();
  // clip: [v]_+
  return std::max(v, 0.0);
}

// NOTE: Gradient includes all smooth terms (typically just the loss only),
// and not the non-smooth regularizers (e.g. L1 and OSCAR),
// but it could include smooth structured regularizers, etc.
void OscarProximalStep(
    const double C1, // constant for L1 regularizer
    const double C_inf, // constant for pairwise L_inf regularizer
    const vector<double>& a,
    vector<double>* z) { // parallel with a upon return

  assert(z != NULL);

  size_t N = a.size();
  assert(z->size() == N);

  // alg2, line 2: initialize groups
  // home for all groups -- destroyed via scope
  // these groups will be mutated over the life of the function
  vector<OscarGroup> groups;
  groups.resize(N);
  for (size_t i = 0; i < N; i++) {
    groups[i]._orig_indices.push_back(i);
    groups[i]._init_value = a.at(i);
  }

  // alg2, line 1: sort in decreasing order
  std::sort(groups.begin(), groups.end(),
            [](const OscarGroup& a, const OscarGroup& b) { return a._init_value > b._init_value; });

  for (size_t i = 0; i < N; i++) {
    groups[i]._sorted_indices.push_back(i);
  }

  // alg2, line 5:  initialize the stack
  // pointers on the stack are owned by the vector "groups"
  stack<OscarGroup*> stack;
  stack.push(&groups.at(0));

  for (size_t i = 1; i < N; i++) {
    OscarGroup& cur_group = groups.at(i);
    while (!stack.empty() && ComputeCommonValue(a, C1, C_inf, N, cur_group) >= ComputeCommonValue(a, C1, C_inf, N, *stack.top())) {
      OscarGroup& top = *stack.top();
      cur_group.MergeWith(top);
      stack.pop();
    }
    stack.push(&cur_group);
  }

  // sanity checking guard so that we know we've assigned all weights
  // (in case groups somehow became malformed)
  double GUARD_VALUE = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < N; i++) {
    (*z)[i] = GUARD_VALUE;
  }

  int iGroup = 0;
  while (!stack.empty()) {
    const OscarGroup& group = *stack.top();
    stack.pop();
    double common_weight = ComputeCommonValue(a, C1, C_inf, N, group);
    
    cerr << "Group " << iGroup++ << ": " << common_weight << endl;
    for (size_t idx : group._orig_indices) {
      (*z)[idx] = common_weight;
    }
  }

  // sanity check for post condition that all z's should have been set
  for (size_t i = 0; i < N; i++) {
    assert(z->at(i) != GUARD_VALUE);
  }
}

// combines the proximal function defined by Zhong & Kwok 2012's algorithm 2
// with the AdaGrad update strategy (which can be thought of as an online accelerated gradient method)
// even though AdaGrad is typically an online optimizer, we use it as a batch optimizer here
void OscarAdaGradOptimize(
    const double C1, // constant for L1 regularizer
    const double C_inf, // constant for pairwise L_inf regularizer
    const double init_learning_rate,
    const double nonadapted_learning_rate,
    const int buffer_size, // number of historical gradients to consider when adapting the learning rate (use -1 for standard adagrad)
    const int iterations,
    const vector<double>& init_weights,
    vector<double>* learned_weights,
    std::function<void (const vector<double>, vector<double>*)> gradient_func) {

  assert(learned_weights != NULL);
  assert(init_learning_rate > 0.0);

  size_t N = init_weights.size();

  // running sum (over iterations) of unnormalized gradient squares
  vector<double> G;
  G.resize(N);

  // last N gradients so that we can be "forgetful" in our adaptive learning rate
  // so that very large updates early on don't prevent us from making appropriately sized updates later
  // TODO: Switch this to a boost circular buffer
  queue<vector<double>> g_recent;

  // gradient for smooth portion of the objective for the current iteration
  vector<double> gradient;
  gradient.resize(N);

  // inputs for dual optimization problem of the proximal step
  vector<double> a;
  a.resize(N);

  // outputs of dual optimization problem of the proximal step
  vector<double> z;
  z.resize(N);

  vector<double> cur_weights = init_weights;

  // this vector is populated via a standard AdaGrad update
  // and then handed to PrepareProximalInputs before running the proximal step
  vector<double> pre_proximal_weights;
  pre_proximal_weights.resize(N);

  for (int i = 0; i < iterations; i++) {
    cerr << "AdaGradOscar iteration " << i << endl;
    // TODO: We could use mini-batches here instead of full batch optimization

    // TOOD: The gradient here could be a sparse vector
    // if using fairly small minibatch sizes for SGD
    gradient_func(cur_weights, &gradient);

    // update the running sum of squared gradients used for the adaptive learning rate
    if (buffer_size > 0) {
      for (size_t j = 0; j < N; j++) {
        G[j] += gradient.at(j) * gradient.at(j);
      }
    }

    // "forget" the oldest gradient for the adaptive learning rate iff the buffer is full
    if (buffer_size >= 0) {
      if ((int)g_recent.size() == buffer_size) {
        const vector<double>& oldest_g = g_recent.front();
        for (size_t j = 0; j < N; j++) {
          G[j] -= oldest_g.at(j) * oldest_g.at(j);
        }      
        g_recent.pop();
      }
      g_recent.push(gradient); // note: copy
    }

    for (size_t i = 0; i < N; i++) {
      // TODO: We don't actually need to take the sqrt each time since
      // for sparse gradients, most elements of G don't actually change between time steps
      double effective_rate = EffectiveLearningRate(init_learning_rate, nonadapted_learning_rate, G.at(i));
      cerr << "EffectiveRate_" << i << ": " << effective_rate << " G: " << G.at(i) << endl;
      pre_proximal_weights[i] = abs(cur_weights.at(i) + effective_rate * gradient.at(i));
    }

    for (size_t i = 0; i < a.size(); i++) {
      cerr << "ppw_" << i << " = " << pre_proximal_weights.at(i) << endl;
    }
    
    OscarPrepareProximalInputs(pre_proximal_weights, &a);

    for (size_t i = 0; i < a.size(); i++) {
      cerr << "a_" << i << " = " << a.at(i) << endl;
    }

    OscarProximalStep(C1, C_inf, a, &z);

    for (size_t i = 0; i < z.size(); i++) {
      cerr << "z_" << i << " = " << z.at(i) << endl;
    }

    // TODO: Avoid copy?
    vector<double> new_weights;
    new_weights.resize(N);
    OscarWeightsFromProximalSolution(z, pre_proximal_weights, &new_weights);

    for (size_t i = 0; i < new_weights.size(); i++) {
      cerr << "NewW_" << i << " = " << new_weights.at(i) << endl;
    }

    cur_weights = new_weights;
  }

  *learned_weights = cur_weights;
}
