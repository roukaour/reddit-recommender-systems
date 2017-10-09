#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>
#include <ctime>

typedef double Statistic;
typedef unsigned int Count;
typedef unsigned short ID; // 65,536 IDs; enough for 43,976 users or 11,675 subreddits, not 3,436,063 posts
typedef float Affinity; // 32-bit precision; enough for 824,521 user-subreddit affinities
typedef float Affinity_Diff;
typedef std::tuple<Affinity, Count, ID> Datum;
typedef std::unordered_map<ID, Datum> Data_Sparse_Vector;
typedef std::unordered_map<ID, Data_Sparse_Vector> Data_Sparse_Matrix;
typedef std::vector<Affinity> Affinities_Dense_Vector;
typedef std::vector<Statistic> Correlations_Dense_Matrix;
typedef std::vector<Statistic> Statistic_Vector;
typedef std::vector<Statistic_Vector> Statistic_Matrix;
typedef std::pair<Affinity, Affinity> Result;
typedef std::vector<Result> Ranked_Results;

enum { AFFINITY = 0, COUNT = 1, SAMPLE = 2 };

std::istream &operator>>(std::istream &in, Datum &p) {
	return in >> std::get<AFFINITY>(p) >> std::get<COUNT>(p) >> std::get<SAMPLE>(p);
}

Statistic seconds_between(clock_t start, clock_t end) {
	return (Statistic)(end - start) / CLOCKS_PER_SEC;
}

class Recommender {
protected:
	ID _test_sample_id;
public:
	inline Recommender(ID test_sample_id) : _test_sample_id(test_sample_id) {}
	virtual ~Recommender() {}
	virtual std::string name(void) const = 0;
	virtual std::string description(void) const = 0;
	virtual Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const = 0;
	void evaluate(void);
};

class Worst_Recommender : public Recommender {
public:
	inline Worst_Recommender(ID test_sample_id) : Recommender(test_sample_id) {}
	inline std::string name(void) const { return "Worst"; }
	inline std::string description(void) const { return "worst possible"; }
	inline Affinity predict_affinity(ID, ID, Affinity actual_affinity) const { return actual_affinity < 0.5f ? 1.0f : 0.0f; }
};

class Random_Recommender : public Recommender {
public:
	inline Random_Recommender(ID test_sample_id) : Recommender(test_sample_id) {}
	inline std::string name(void) const { return "Random"; }
	inline std::string description(void) const { return "random baseline"; }
	inline Affinity predict_affinity(ID, ID, Affinity) const { return (Affinity)rand() / RAND_MAX; }
};

class Affinity_Average_Recommender : public Recommender {
public:
	inline Affinity_Average_Recommender(ID test_sample_id) : Recommender(test_sample_id) {}
	inline std::string name(void) const { return "Affinity-average"; }
	inline std::string description(void) const { return "overall average affinity"; }
	Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const;
};

class User_Average_Recommender : public Recommender {
public:
	inline User_Average_Recommender(ID test_sample_id) : Recommender(test_sample_id) {}
	inline std::string name(void) const { return "User-average"; }
	inline std::string description(void) const { return "users' average affinities"; }
	Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const;
};

class Subreddit_Average_Recommender : public Recommender {
public:
	inline Subreddit_Average_Recommender(ID test_sample_id) : Recommender(test_sample_id) {}
	inline std::string name(void) const { return "Subreddit-average"; }
	inline std::string description(void) const { return "subreddits' average affinities"; }
	Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const;
};

class Combined_Average_Recommender : public Recommender {
private:
	Statistic _user_fraction;
public:
	inline Combined_Average_Recommender(Statistic user_fraction, ID test_sample_id) : Recommender(test_sample_id), _user_fraction(user_fraction) {}
	std::string name(void) const;
	std::string description(void) const;
	Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const;
};

class User_Based_CF_Recommender : public Recommender {
private:
	Statistic _rho;
	bool _damping;
	Correlations_Dense_Matrix _user_user_correlations;
public:
	User_Based_CF_Recommender(Statistic rho, bool damping, ID test_sample_id);
	std::string name(void) const;
	std::string description(void) const;
	Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const;
};

class Subreddit_Based_CF_Recommender : public Recommender {
private:
	Statistic _rho;
	bool _damping;
	Correlations_Dense_Matrix _sr_sr_correlations;
public:
	Subreddit_Based_CF_Recommender(Statistic rho, bool damping, ID test_sample_id);
	std::string name(void) const;
	std::string description(void) const;
	Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const;
};

class SVD_Recommender : public Recommender {
private:
	Count _k;
	bool _pure;
	Statistic_Matrix _svd_matrix_vt, _svd_matrix_usqrtsigma, _svd_matrix_sqrtsigmavt;
public:
	SVD_Recommender(ID test_sample_id);
	std::string name(void) const;
	std::string description(void) const;
	inline void setup(Count k, bool pure) { _k = k; _pure = pure; }
	Affinity predict_affinity(ID active_user_id, ID active_sr_id, Affinity actual_affinity) const;
};
