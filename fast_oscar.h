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

// zero-safe implementation of C99's copysign
// (floats are signed, which can lead to confusing results)
template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

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

// combines the proximal function defined by Zhong & Kwok 2012's algorithm 2
// with the AdaGrad update strategy (which can be thought of as an online accelerated gradient method)
// even though AdaGrad is typically an online optimizer, we use it as a batch optimizer here
class AdaGradOscarOptimizer {

  double EffectiveLearningRate(const size_t i) const {
    double sqrt_G = sqrt(_G.at(i));
    if (sqrt_G > 0.0)
      return _nonadapted_learning_rate + _init_learning_rate / sqrt_G;
    else
      return _nonadapted_learning_rate;
  }

  // prepare inputs for OSCAR proximal step, according to the diagonal AdaGrad update strategy
  void OscarPrepareProximalInputs(const vector<double>& pre_proximal_weights, vector<double>* a) const {
    assert(a != NULL);
    assert(a->size() == _N);
    assert(pre_proximal_weights.size() == _N);
    for (size_t i = 0; i < _N; i++) {
      (*a)[i] = abs(pre_proximal_weights.at(i));
    }
  }

  void OscarWeightsFromProximalSolution(const vector<double>& z, const vector<double>& pre_proximal_weights, vector<double>* new_weights) const {
    assert (new_weights != NULL);
    assert (z.size() == _N);
    assert (z.size() == _N);
    
    for (size_t i = 0; i < _N; i++) {
      int polarity = sgn(pre_proximal_weights.at(i));
      if (_verbose) {
	cerr << "Polarity_" << i << " " << polarity << "; " << pre_proximal_weights.at(i) << endl;
      }
      (*new_weights)[i] = polarity * z.at(i);
    }
  }

  // w: Measures contribution of regularizers
  // used for determining an actual or hypothetical common parameter value within a group
  // i is the index of interest **in the sorted list** of a's
  double ComputeW(const size_t i) const {
    //const double C1, // constant for L1 regularizer
    //const double C_inf, // constant for pairwise L_inf regularizer
    //const size_t N, // dimensionality / feature count
    return _C1 + _C_inf * (_N - i);
  }

  // compute the common value for a group or proposed group
  // equation (15)
  double ComputeCommonValue(const vector<double>& a, const OscarGroup& group) const {
    //const double C1, // constant for L1 regularizer
    //const double C_inf, // constant for pairwise L_inf regularizer
    //const size_t N, // dimensionality / feature count

    double numerator = 0.0;
    for (size_t k = 0; k < group._sorted_indices.size(); k++) {
      size_t orig_idx = group._orig_indices.at(k); // for accessing a's, etc.
      size_t sorted_idx = group._sorted_indices.at(k); // for determining impact under pairwise L_inf regularizer
      assert(orig_idx < _N);
      assert(sorted_idx < _N);
      numerator += a.at(orig_idx) - ComputeW(sorted_idx);
    }
    double v =  numerator / group.Size();
    // clip: [v]_+
    return std::max(v, 0.0);
  }

  // NOTE: Gradient includes all smooth terms (typically just the loss only),
  // and not the non-smooth regularizers (e.g. L1 and OSCAR),
  // but it could include smooth structured regularizers, etc.
  void OscarProximalStep(const vector<double>& a, vector<double>* z) const {
    assert(z != NULL);
    assert(a.size() == _N);
    assert(z->size() == _N);

    // alg2, line 2: initialize groups
    // home for all groups -- destroyed via scope
    // these groups will be mutated over the life of the function
    vector<OscarGroup> groups;
    groups.resize(_N);
    for (size_t i = 0; i < _N; i++) {
      groups[i]._orig_indices.push_back(i);
      groups[i]._init_value = a.at(i);
    }

    // alg2, line 1: sort in decreasing order
    std::sort(groups.begin(), groups.end(),
	      [](const OscarGroup& a, const OscarGroup& b) { return a._init_value > b._init_value; });
    
    for (size_t i = 0; i < _N; i++) {
      groups[i]._sorted_indices.push_back(i);
    }

    // alg2, line 5:  initialize the stack
    // pointers on the stack are owned by the vector "groups"
    stack<OscarGroup*> stack;
    stack.push(&groups.at(0));

    for (size_t i = 1; i < _N; i++) {
      OscarGroup& cur_group = groups.at(i);
      while (!stack.empty() && ComputeCommonValue(a, cur_group) >= ComputeCommonValue(a, *stack.top())) {
	OscarGroup& top = *stack.top();
	cur_group.MergeWith(top);
	stack.pop();
      }
      stack.push(&cur_group);
    }
    
    // sanity checking guard so that we know we've assigned all weights
    // (in case groups somehow became malformed)
    const double GUARD_VALUE = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < _N; i++) {
      (*z)[i] = GUARD_VALUE;
    }
    
    int iGroup = 0;
    while (!stack.empty()) {
      const OscarGroup& group = *stack.top();
      stack.pop();
      double common_weight = ComputeCommonValue(a, group);
      
      if (_verbose) cerr << "Group " << iGroup++ << ": " << common_weight << endl;
      for (size_t idx : group._orig_indices) {
	(*z)[idx] = common_weight;
      }
    }
    
    // sanity check for post condition that all z's should have been set
    for (size_t i = 0; i < _N; i++) {
      assert(z->at(i) != GUARD_VALUE);
    }
  }

  const double _C1;
  const double _C_inf;
  const double _init_learning_rate;
  const double _nonadapted_learning_rate;
  const int _buffer_size;
  const int _iterations;
  const size_t _N;
  vector<double> _prev_weights;
  const bool _verbose;

  vector<double> _G;   // running sum (over iterations) of unnormalized gradient squares
  
  // last N gradients so that we can be "forgetful" in our adaptive learning rate
  // so that very large updates early on don't prevent us from making appropriately sized updates later
  // TODO: Switch this to a boost circular buffer
  queue<vector<double>> _g_recent;
  
  // tempoarary variable: inputs for dual optimization problem of the proximal step
  vector<double> _a;
  
  // tempoarary variable: outputs of dual optimization problem of the proximal step
  vector<double> _z;
  
  // this vector is populated via a standard AdaGrad update
  // and then handed to PrepareProximalInputs before running the proximal step
  vector<double> _pre_proximal_weights;
  
  int _cur_iteration;
  
  // standard optimizer interface
  public:
  AdaGradOscarOptimizer(
    const double C1, // constant for L1 regularizer
    const double C_inf, // constant for pairwise L_inf regularizer
    const double init_learning_rate,
    const double nonadapted_learning_rate,
    const int buffer_size, // number of historical gradients to consider when adapting the learning rate (use -1 for standard adagrad)
    const int iterations,
    const vector<double>& init_weights,
    const bool verbose)
   : _C1(C1),
     _C_inf(C_inf),
     _init_learning_rate(init_learning_rate),
     _nonadapted_learning_rate(nonadapted_learning_rate),
     _buffer_size(buffer_size),
     _iterations(iterations),
     _N(init_weights.size()),
     _prev_weights(init_weights),
     _verbose(verbose),
     // initialize non-parameters
     _cur_iteration(0)
   {
     assert(_C1 >= 0.0);
     assert(_C_inf >= 0.0);
     assert(_init_learning_rate >= 0.0);
     assert(_nonadapted_learning_rate >= 0.0);
     assert(_buffer_size >= -1);
     assert(_iterations >= 0);

     _G.resize(_N);
     _pre_proximal_weights.resize(_N);
     _a.resize(_N);
     _z.resize(_N);
   }

  // TODO: The gradient here could be a sparse vector
  // if using fairly small minibatch sizes for SGD
  void Optimize(double /*cll*/, const vector<double>& gradient, vector<double>* updated_weights) {
   assert(updated_weights != NULL);
   assert(updated_weights->size() == _N);
   assert(gradient.size() == _N);

   if (_verbose) cerr << "AdaGradOscar iteration " << _cur_iteration << endl;
   // TODO: We could use mini-batches here instead of full batch optimization
   
   // update the running sum of squared gradients used for the adaptive learning rate
   if (_buffer_size > 0) {
     for (size_t j = 0; j < _N; j++) {
       _G[j] += gradient.at(j) * gradient.at(j);
     }
   }
   
   // "forget" the oldest gradient for the adaptive learning rate iff the buffer is full
   if (_buffer_size >= 0) {
     if ((int)_g_recent.size() == _buffer_size) {
       const vector<double>& oldest_g = _g_recent.front();
       for (size_t j = 0; j < _N; j++) {
	 _G[j] -= oldest_g.at(j) * oldest_g.at(j);
       }      
       _g_recent.pop();
     }
     _g_recent.push(gradient); // note: copy
   }
   
   for (size_t i = 0; i < _N; i++) {
     // TODO: We don't actually need to take the sqrt each time since
     // for sparse gradients, most elements of G don't actually change between time steps
     double effective_rate = EffectiveLearningRate(i);
     if (_verbose) cerr << "EffectiveRate_" << i << ": " << effective_rate << " G: " << _G.at(i) << endl;
     _pre_proximal_weights[i] = abs(_prev_weights.at(i) + effective_rate * gradient.at(i));
   }
   
   if (_verbose) {
     for (size_t i = 0; i < _N; i++) {
       cerr << "ppw_" << i << " = " << _pre_proximal_weights.at(i) << endl;
     }
   }
   
   OscarPrepareProximalInputs(_pre_proximal_weights, &_a);
   
   if (_verbose) {
     for (size_t i = 0; i < _N; i++) {
       cerr << "a_" << i << " = " << _a.at(i) << endl;
     }
   }
   
   OscarProximalStep(_a, &_z);
   
   if (_verbose) {
     for (size_t i = 0; i < _N; i++) {
       cerr << "z_" << i << " = " << _z.at(i) << endl;
     }
   }
   
   OscarWeightsFromProximalSolution(_z, _pre_proximal_weights, &_prev_weights);
   
   if (_verbose) {
     for (size_t i = 0; i < _N; i++) {
       cerr << "NewW_" << i << " = " << _prev_weights.at(i) << endl;
     }
   }
   
   _cur_iteration++;
   *updated_weights = _prev_weights; // element-wise copy
 }

 bool HasConverged() const {
   assert(_cur_iteration <= _iterations);
   return (_cur_iteration == _iterations);
 }
};
