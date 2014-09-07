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
    assert(_G.at(i) >= 0.0);
    double sqrt_G = sqrt(_G.at(i));
    assert(sqrt_G >= 0.0);

    double result;
    if (sqrt_G != 0.0) {
      result = _nonadapted_learning_rate + _init_learning_rate / sqrt_G;
    } else {
      result = _nonadapted_learning_rate;
    }
    assert(std::isfinite(result));
    return result;
  }

  // prepare inputs for OSCAR proximal step, according to the diagonal AdaGrad update strategy
  void OscarPrepareProximalInputs(const vector<double>& pre_proximal_weights, vector<double>* a) const {
    assert(a != NULL);
    assert(a->size() == _N);
    assert(pre_proximal_weights.size() == _N);
    for (size_t i = 0; i < _N; i++) {
      if (_oscar_feats.at(i)) {
	(*a)[i] = abs(pre_proximal_weights.at(i));
      } else {
	(*a)[i] = 0;
      }
      assert(std::isfinite(a->at(i)));
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
      if (_oscar_feats.at(i)) {
	(*new_weights)[i] = polarity * z.at(i);
      } else {
	// not an oscar feature -- just return the pre_proximal_weight, which
	// is the result of applying a vanilla adagrad update
	(*new_weights)[i] = pre_proximal_weights.at(i);
      }
      assert(std::isfinite(new_weights->at(i)));
    }
  }

  // w: Measures contribution of regularizers
  // used for determining an actual or hypothetical common parameter value within a group
  // i is the index of interest **in the sorted list** of a's
  double ComputeW(const size_t i) const {
    assert(i < _N);
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
      assert(std::isfinite(numerator));
    }
    double v =  numerator / group.Size();
    assert(std::isfinite(v));
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
    int num_active = 0;
    int num_dof = 0;
    while (!stack.empty()) {
      const OscarGroup& group = *stack.top();
      stack.pop();
      double common_weight = ComputeCommonValue(a, group);
      
      if (_verbose) cerr << "Group_" << iGroup << ": " << common_weight << " Size: " << group.Size() << endl;
      for (size_t idx : group._orig_indices) {
	(*z)[idx] = common_weight;
      }

      if (common_weight != 0)
      {
        num_active += group.Size();
        num_dof++;
      }
        
      iGroup++;
    }
    cerr << "FastOSCAR: Active features: " << num_active << " Degrees of freedom: " << num_dof << endl;
    
    // sanity check for post condition that all z's should have been set
    for (size_t i = 0; i < _N; i++) {
      assert(z->at(i) != GUARD_VALUE);
      assert(std::isfinite(z->at(i)));
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
  vector<bool> _oscar_feats;
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
    const vector<bool>& oscar_feats, // true for features that we should apply OSCAR to (others will remain untouched and will never be regularized away by oscar)
    const bool verbose)
   : _C1(C1),
     _C_inf(C_inf),
     _init_learning_rate(init_learning_rate),
     _nonadapted_learning_rate(nonadapted_learning_rate),
     _buffer_size(buffer_size),
     _iterations(iterations),
     _N(init_weights.size()),
     _prev_weights(init_weights),
     _oscar_feats(oscar_feats),
     _verbose(verbose),

     // initialize non-parameters
     _G(_N, 0.0),
     _a(_N, 0.0),
     _z(_N, 0.0),
     _pre_proximal_weights(_N, 0.0),
     _cur_iteration(0)
   {
     assert(_C1 >= 0.0);
     assert(_C_inf >= 0.0);
     assert(_init_learning_rate >= 0.0);
     assert(_nonadapted_learning_rate >= 0.0);
     assert(_buffer_size >= -1);
     assert(_iterations >= 0);
   }

  // TODO: The gradient here could be a sparse vector
  // if using fairly small minibatch sizes for SGD
  void Optimize(double /*cll*/, const vector<double>& gradient_noreg, const vector<double>& gradient, vector<double>* updated_weights) {
   assert(updated_weights != NULL);
   assert(updated_weights->size() == _N);
   assert(gradient_noreg.size() == _N);
   assert(gradient.size() == _N);

   // TODO: We could use mini-batches here instead of full batch optimization
   
   // update the running sum of squared gradients used for the adaptive learning rate
   if (_buffer_size != 0) {
     for (size_t j = 0; j < _N; j++) {
       _G[j] += gradient.at(j) * gradient.at(j);
       assert(std::isfinite(_G.at(j)));
       assert(_G.at(j) >= 0.0);
     }
   }
   
   // "forget" the oldest gradient for the adaptive learning rate iff the buffer is full
   if (_buffer_size > 0) {
     if ((int)_g_recent.size() == _buffer_size) {
       const vector<double>& oldest_g = _g_recent.front();
       for (size_t j = 0; j < _N; j++) {
	 double oldest_g_sq = oldest_g.at(j) * oldest_g.at(j);
	 double updated_value = _G.at(j) - oldest_g_sq;
	 assert(std::isfinite(_G.at(j)));
	 assert(std::isfinite(oldest_g_sq));
	 assert(oldest_g_sq >= 0.0);
	 assert(_G.at(j) >= 0.0);
	 assert(updated_value >= -1e-6);

	 if (_verbose) cerr << "G_" << j << ": " << _G.at(j) << " oldest_g_sq: " << oldest_g_sq << endl;
	 if (updated_value < 0.0) // in case of floating point error
	   _G[j] = 0.0;
	 else
	   _G[j] = updated_value;

	 assert(std::isfinite(_G.at(j)));
	 assert(_G.at(j) >= 0.0);
       }      
       _g_recent.pop();
     }
     _g_recent.push(gradient); // note: copy
   }
   
   for (size_t i = 0; i < _N; i++) {
     // TODO: We don't actually need to take the sqrt each time since
     // for sparse gradients, most elements of G don't actually change between time steps
     double effective_rate = EffectiveLearningRate(i);
     assert(std::isfinite(_prev_weights.at(i)));
     assert(std::isfinite(gradient.at(i)));
     assert(std::isfinite(effective_rate));
     _pre_proximal_weights[i] = _prev_weights.at(i) - effective_rate * gradient.at(i);
     assert(std::isfinite(_pre_proximal_weights.at(i)));
   }
   
   OscarPrepareProximalInputs(_pre_proximal_weights, &_a);
   OscarProximalStep(_a, &_z);   
   OscarWeightsFromProximalSolution(_z, _pre_proximal_weights, updated_weights);

   if (_verbose) {
     cerr << "AdaGradOscar iteration " << _cur_iteration << endl;
     for (size_t i = 0; i < _N; i++) {
       cerr << "PrevW_" << i << " = " << _prev_weights.at(i) << endl;
       cerr << "gradient_noreg_" << i << " = " << gradient_noreg.at(i) << endl;
       cerr << "gradient_" << i << " = " << gradient.at(i) << endl;
       cerr << "EffectiveRate_" << i << ": " << EffectiveLearningRate(i) << " G=sum over g^2: " << _G.at(i) << endl;
       cerr << "ppw_" << i << " = " << _pre_proximal_weights.at(i) << endl;
       cerr << "a_" << i << " = " << _a.at(i) << endl;
       cerr << "z_" << i << " = " << _z.at(i) << endl;
       cerr << "NewW_" << i << " = " << updated_weights->at(i) << endl;
     }
   }
   
   _cur_iteration++;
   _prev_weights = *updated_weights;
 }

 bool HasConverged() const {
   assert(_cur_iteration <= _iterations);
   return (_cur_iteration == _iterations);
 }
};
