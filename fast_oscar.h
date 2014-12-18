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
  size_t _orig_idx;
  size_t _sorted_idx;
  double _init_value;

  // separate the numerator into contributions from categories:
  // * a (the absolute value of the pre-proximal weight)
  // * L1 (the contributions from the L1 regularizer)
  // * Linf (the contributions from the Linf regularizer)
  double _numerator_a;
  double _numerator_L1;
  double _numerator_Linf;

  double _a_contrib;
  double _L1_contrib;
  double _Linf_contrib;

  size_t _n;

  OscarGroup() {
    _numerator_a = 0.0;
    _numerator_L1 = 0.0;
    _numerator_Linf = 0.0;

    _a_contrib = 0.0;
    _L1_contrib = 0.0;
    _Linf_contrib = 0.0;
    _n = 0;
  }

  size_t Size() const {
    return _n;
  }

  // initialize with a single point
  void Init(size_t orig_idx, double init_value, double C1) {
    _orig_idx = orig_idx;
    _init_value = init_value; // a.at(orig_idx)
    _n = 1;
    
    _numerator_a = init_value;
    _numerator_L1 = -C1;

    // this begins equal to the numerator since we divide by one
    _a_contrib = _numerator_a;
    _L1_contrib = _numerator_L1;
  }
  void SetSortedIndex(size_t sorted_idx, size_t oscar_feat_count, double C_inf) {
    _sorted_idx = sorted_idx;
    _numerator_Linf = -ComputeW_inf(sorted_idx, oscar_feat_count, C_inf);

    // this begins equal to the numerator since we divide by one
    _Linf_contrib = _numerator_Linf;
  }

  void MergeWith(OscarGroup& that) {
    // TODO: Can we get away without these altogether?
    _n += that._n;

    _numerator_a += that._numerator_a;
    _numerator_L1 += that._numerator_L1;
    _numerator_Linf += that._numerator_Linf;
    assert(std::isfinite(_numerator_a));
    assert(std::isfinite(_numerator_L1));
    assert(std::isfinite(_numerator_Linf));

    _a_contrib = _numerator_a / _n;
    _L1_contrib = _numerator_L1 / _n;
    _Linf_contrib = _numerator_Linf / _n;
  }
  
  void Clear() {
    _n = 0;
  }

  // w: Measures contribution of the L_inf regularizer
  // used for determining an actual or hypothetical common parameter value within a group
  // i is the index of interest **in the sorted list** of a's
  // (see text just below Eq 13)
  inline double ComputeW_inf(const size_t i,
                             const size_t oscar_feat_count,
                             const double C_inf) const {
    assert(i < oscar_feat_count);
    //const double C1, // constant for L1 regularizer
    //const double C_inf, // constant for pairwise L_inf regularizer
    //const size_t N, // dimensionality / feature count
    return C_inf * (oscar_feat_count - i);
  }

  // compute the common value for a group or proposed group
  // equation (15)
  // optionally return individual contributions from regularizers and pre-proximal weight mass for debuggging
  double ComputeCommonValue(const bool include_Linf) { // include Linf in returned value? this allows us to use it for grouping, but not for final weight value calculation

    // clip: [v]_+ (only include L1)
    double clipped = std::max(_a_contrib + _L1_contrib, 0.0);
    assert(std::isfinite(clipped));
    double common_value;
    if (clipped == 0.0) {
      common_value = 0.0;
    } else {
      // if we survived clipping via L1, allow Linf to change our ranking, but
      // not clip us (this gives a non-octagonal regularizer, but that's okay)
      if (include_Linf) {
        common_value = clipped + _Linf_contrib;
        assert(std::isfinite(common_value));
      } else {
        common_value = clipped;
      }
    }
    return common_value;
  }

};

// combines the proximal function defined by Zhong & Kwok 2012's algorithm 2
// with the AdaGrad update strategy (which can be thought of as an online accelerated gradient method)
// even though AdaGrad is typically an online optimizer, we use it as a batch optimizer here
class AdaGradOscarOptimizer {

  double EffectiveLearningRate(const size_t i) const {
    assert(i < _N);
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
    assert(a->size() == _oscar_feat_count);
    assert(pre_proximal_weights.size() == _N);
    assert(_oscar_feats.size() == _N);
    assert(_oscar_feat_count <= _N);

    size_t j = 0;
    for (size_t i = 0; i < _N; ++i) {
      if (_oscar_feats.at(i)) {
	(*a)[j] = abs(pre_proximal_weights.at(i));
        assert(std::isfinite(a->at(j)));
        ++j;
      }
    }
    assert(j == a->size());
  }

  void OscarWeightsFromProximalSolution(const vector<double>& z, const vector<double>& pre_proximal_weights, vector<double>* new_weights) const {
    assert(new_weights != NULL);
    assert(z.size() == _oscar_feat_count);
    assert(pre_proximal_weights.size() == _N);
    assert(new_weights->size() == _N);
    assert(_oscar_feat_count <= _N);
    
    size_t j = 0;
    for (size_t i = 0; i < _N; ++i) {
      int polarity = sgn(pre_proximal_weights.at(i));
      if (_verbose) {
	cerr << "Polarity_" << i << " " << polarity << "; " << pre_proximal_weights.at(i) << endl;
      }
      if (_oscar_feats.at(i)) {
	(*new_weights)[i] = polarity * z.at(j);
        ++j;
      } else {
	// not an oscar feature -- just return the pre_proximal_weight, which
	// is the result of applying a vanilla adagrad update
	(*new_weights)[i] = pre_proximal_weights.at(i);
      }
      assert(std::isfinite(new_weights->at(i)));
    }
    assert(j == z.size());
  }

  // NOTE: Gradient includes all smooth terms (typically just the loss only),
  // and not the non-smooth regularizers (e.g. L1 and OSCAR),
  // but it could include smooth structured regularizers, etc.
  //
  // the pre-proximal weights are used only for calculating the number of active features
  // since there may be non-OSCAR features
  void OscarProximalStep(const vector<double>& pre_proximal_weights,
                         const vector<double>& a,
			 vector<double>* z) const {
    assert(z != NULL);
    assert(pre_proximal_weights.size() == _N);
    assert(a.size() == _oscar_feat_count);
    assert(z->size() == _oscar_feat_count);
    assert(_oscar_feat_count <= _N);

    std::cerr << "FastOSCAR: INIT ";

    // alg2, line 2: initialize groups
    // home for all groups -- destroyed via scope
    // these groups will be mutated over the life of the function
    vector<OscarGroup> groups;
    groups.resize(_oscar_feat_count);
    for (size_t i = 0; i < _oscar_feat_count; ++i) {
      groups[i].Init(i, a.at(i), _C1);
    }

    // alg2, line 1: sort in decreasing order
    std::cerr << "SORT ";
    std::sort(groups.begin(), groups.end(),
	      [](const OscarGroup& a, const OscarGroup& b) { return a._init_value > b._init_value; });

    // create a mapping between sorted feature indices and original indices
    // so that we don't need to keep any intermediate vectors
    // we'll keep a start and end index into the sorted array in each group
    std::cerr << "MAP ";
    vector<size_t> sorted_idx_to_orig_idx;
    sorted_idx_to_orig_idx.resize(_oscar_feat_count);

    for (size_t i = 0; i < _oscar_feat_count; ++i) {
      sorted_idx_to_orig_idx[i] = groups[i]._orig_idx;

      groups[i].SetSortedIndex(i, _oscar_feat_count, _C_inf);
    }

    // alg2, line 5:  initialize the stack
    // pointers on the stack are owned by the vector "groups"
    stack<OscarGroup*> stack;
    stack.push(&groups.at(0));

    // in this outer loop, we determine how many groups we will have
    std::cerr << "GROUPING ";
    assert(_oscar_feat_count <= _N);
    for (size_t i = 1; i < _oscar_feat_count; i++) {
      OscarGroup& cur_group = groups.at(i);
      bool done_with_group = false;
      while (!stack.empty() && !done_with_group) {
        OscarGroup& next_group = *stack.top();
        double cur_common_value = cur_group.ComputeCommonValue(true);
        double next_common_value = next_group.ComputeCommonValue(true);

        if (_debug_proximal_step) {
          cerr << "i=" << i << "; ";
          if (cur_common_value >= next_common_value) {
            cerr << "MERGED ";
          } else {
            cerr << "NOT MERGED ";
          }

          double delta_Linf_contrib = next_group._Linf_contrib - cur_group._Linf_contrib;
          cerr << "cur_group " << cur_common_value << " (a=" << cur_group._a_contrib << "; L1=" << cur_group._L1_contrib << "; Linf=" << cur_group._Linf_contrib << ") "
               << "with (>=?) next_group " << next_common_value << " (a=" << next_group._a_contrib << "; L1=" << next_group._L1_contrib << "; Linf=" << next_group._Linf_contrib << ") "
               << " -- delta Linf_contrib: " << delta_Linf_contrib
               << " -- group size: " << cur_group.Size();
          if (done_with_group) {
            cerr << " (DONE with group)" << endl;
          } else {
            cerr << " (CONTINUE with group)" << endl;
          }
        }

        if (cur_common_value >= next_common_value) {
          cur_group.MergeWith(next_group);
          next_group.Clear();
          stack.pop(); // pop off next_group aka stack.top()
          // merge and continue looking for merges with this group
        } else {
          // don't merge and stop looking for merges with this group
          done_with_group = true;
        }
      }
      stack.push(&cur_group);
    }
    
    // sanity checking guard so that we know we've assigned all weights
    // (in case groups somehow became malformed)
    const double GUARD_VALUE = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < _oscar_feat_count; i++) {
      (*z)[i] = GUARD_VALUE;
    }
    
    // at this point, we have determined how many groups (degrees of freedom)
    // we will have. we just need to assign a weight to each group.
    std::cerr << "WEIGHTING ";
    int iGroup = 0;
    int num_active = 0;
    int num_dof = 0;
    while (!stack.empty()) {
      OscarGroup& group = *stack.top();
      stack.pop();
      // don't include the L_inf penalty when computing the weight --
      // it can be very harsh. it's useful for bringing weights close together for clustering,
      // but not so relevant to the final weight (especially the polarity of the final weight)
      double common_weight = group.ComputeCommonValue(false);
      
      if (_verbose) cerr << "Group_" << iGroup << ": " << common_weight << " Size: " << group.Size() << endl;
      for (size_t i = 0; i < group._n; ++i) {
        size_t sorted_idx = group._sorted_idx - i;
        size_t orig_idx = sorted_idx_to_orig_idx.at(sorted_idx);
	(*z)[orig_idx] = common_weight;
      }

      // now that we've computed the common feature weight for all features in this group,
      // tally its impact on active feature count and degrees of freedom
      if (common_weight != 0)
      {
        num_active += group.Size();
        ++num_dof;
      }
        
      ++iGroup;
    }

    std::cerr << "DONE" << std::endl;

    // now add in non-OSCAR features
    int all_active = num_active;
    int all_dof = num_dof;
    for (size_t i = 0; i < _N; ++i) {
      if (!_oscar_feats.at(i) && pre_proximal_weights.at(i) != 0.0) {
	++all_active;
	++all_dof;
      }
    }
    cerr << "FastOSCAR: OSCAR active features: " << num_active
	 << " OSCAR degrees of freedom: " << num_dof
	 << " Total active features: " << all_active
	 << " Total degrees of freedom: " << all_dof
	 << endl;
    
    // sanity check for post condition that all z's should have been set
    for (size_t i = 0; i < _oscar_feat_count; i++) {
      assert(z->at(i) != GUARD_VALUE);
      assert(std::isfinite(z->at(i)));
    }
  }

  const double _C1;
  const double _C_inf;
  const bool _use_Linf_in_weights;
  const double _init_learning_rate;
  const double _nonadapted_learning_rate;
  const int _buffer_size;
  const int _iterations;
  const size_t _N;
  size_t _oscar_feat_count;
  vector<double> _prev_weights;
  vector<bool> _oscar_feats;
  const bool _verbose;
  const bool _debug_proximal_step;

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
    const bool use_Linf_in_weights, // use Linf penalty for computing weight values? (false means use it only for grouping purposes; recommended: true)
    const double init_learning_rate,
    const double nonadapted_learning_rate,
    const int buffer_size, // number of historical gradients to consider when adapting the learning rate (use -1 for standard adagrad)
    const int iterations,
    const vector<double>& init_weights,
    const vector<bool>& oscar_feats, // true for features that we should apply OSCAR to (others will remain untouched and will never be regularized away by oscar)
    const bool verbose)
   : _C1(C1),
     _C_inf(C_inf),
     _use_Linf_in_weights(use_Linf_in_weights),
     _init_learning_rate(init_learning_rate),
     _nonadapted_learning_rate(nonadapted_learning_rate),
     _buffer_size(buffer_size),
     _iterations(iterations),
     _N(init_weights.size()),
     _prev_weights(init_weights),
     _oscar_feats(oscar_feats),
     _verbose(verbose),
     _debug_proximal_step(verbose),

    _oscar_feat_count(std::count_if(oscar_feats.begin(), oscar_feats.end(), [](const bool b) { return b == true; })),

     // initialize non-parameters
     _G(_N, 0.0),
     _a(_oscar_feat_count, 0.0),
     _z(_oscar_feat_count, 0.0),
     _pre_proximal_weights(_N, 0.0),
     _cur_iteration(0)
   {
     assert(_C1 >= 0.0);
     assert(_C_inf >= 0.0);
     assert(_init_learning_rate >= 0.0);
     assert(_nonadapted_learning_rate >= 0.0);
     assert(_buffer_size >= -1);
     assert(_iterations >= 0);
     assert(_oscar_feats.size() >= _N);
     cerr << "Initialized a new adagrad optimizer instance with " << _oscar_feat_count << " OSCAR feats" << endl;
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
   OscarProximalStep(_pre_proximal_weights, _a, &_z);   
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
