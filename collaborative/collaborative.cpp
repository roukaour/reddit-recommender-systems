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
#include <limits>
#include <ctime>

#include "collaborative.h"

std::string g_directory("C:/Users/Remy/Desktop/590/");
std::string g_output_filename("collaborative_output.log");
std::ofstream g_output_file(g_directory + g_output_filename);
std::string g_results_filename("collaborative_results.tsv");
std::ofstream g_results_file(g_directory + g_results_filename);
size_t g_precision = 12;
unsigned int g_random_seed = 42;
Count g_num_samples = 10;
Count g_max_num_singular_values = 500;

Data_Sparse_Matrix g_user_sr_affinities, g_sr_user_affinities;
Count g_num_affinities = 0, g_num_users = 0, g_num_srs = 0;
Affinities_Dense_Vector g_user_average_affinities, g_sr_average_affinities;
float g_average_affinity = 0.853586756913f;

void read_affinities_file() {
	std::string affinities_filename("affinities.tsv");
	std::ifstream affinities_file(g_directory + affinities_filename);
	g_output_file << "Reading " << affinities_filename << "; counting affinities..." << std::endl;
	clock_t start_clock = clock();
	std::string affinities_header;
	std::getline(affinities_file, affinities_header);
	ID user_id, sr_id;
	Datum datum;
	while (affinities_file >> user_id >> sr_id >> datum) {
		g_num_affinities++;
		g_user_sr_affinities[user_id][sr_id] = datum;
		g_sr_user_affinities[sr_id][user_id] = datum;
	}
	g_num_users = (Count)g_user_sr_affinities.size();
	g_num_srs = (Count)g_sr_user_affinities.size();
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done reading " << affinities_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << g_num_affinities << " user-subreddit affinities, between " << g_num_users << " users and " << g_num_srs << " subreddits" << std::endl;
	g_output_file << std::endl;
}

void calc_average_user_affinities(ID test_sample_id) {
	g_user_average_affinities.resize(g_num_users);
	g_output_file << "Calculating users' average affinities..." << std::endl;
	clock_t start_clock = clock();
	for (Data_Sparse_Matrix::const_iterator m_it = g_user_sr_affinities.begin(); m_it != g_user_sr_affinities.end(); ++m_it) {
		ID user_id = (*m_it).first;
		Affinity user_average_affinity = 0.0f;
		Count user_num_affinities = 0;
		const Data_Sparse_Vector &user_affinities = (*m_it).second;
		for (Data_Sparse_Vector::const_iterator v_it = user_affinities.begin(); v_it != user_affinities.end(); ++v_it) {
			ID sample_id = std::get<SAMPLE>((*v_it).second);
			if (test_sample_id < g_num_samples && sample_id != test_sample_id) {
				continue;
			}
			Affinity user_sr_affinity = std::get<AFFINITY>((*v_it).second);
			user_average_affinity += user_sr_affinity;
			user_num_affinities++;
		}
		user_average_affinity = user_num_affinities ? user_average_affinity / user_num_affinities : g_average_affinity;
		g_user_average_affinities[user_id] = user_average_affinity;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done calculating users' average affinities in " << seconds << " seconds" << std::endl;
	g_output_file << std::endl;
}

void calc_average_sr_affinities(ID test_sample_id) {
	g_sr_average_affinities.resize(g_num_srs);
	g_output_file << "Calculating subreddits' average affinities..." << std::endl;
	clock_t start_clock = clock();
	for (Data_Sparse_Matrix::const_iterator m_it = g_sr_user_affinities.begin(); m_it != g_sr_user_affinities.end(); ++m_it) {
		ID sr_id = (*m_it).first;
		Affinity sr_average_affinity = 0.0f;
		Count sr_num_affinities = 0;
		const Data_Sparse_Vector &sr_affinities = (*m_it).second;
		for (Data_Sparse_Vector::const_iterator v_it = sr_affinities.begin(); v_it != sr_affinities.end(); ++v_it) {
			ID sample_id = std::get<SAMPLE>((*v_it).second);
			if (test_sample_id < g_num_samples && sample_id != test_sample_id) {
				continue;
			}
			Affinity sr_user_affinity = std::get<AFFINITY>((*v_it).second);
			sr_average_affinity += sr_user_affinity;
			sr_num_affinities++;
		}
		sr_average_affinity = sr_num_affinities ? sr_average_affinity / sr_num_affinities : g_average_affinity;
		g_sr_average_affinities[sr_id] = sr_average_affinity;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done calculating subreddits' average affinities in " << seconds << " seconds" << std::endl;
	g_output_file << std::endl;
}

bool greater_actual_affinity(Result &a, Result &b) {
	return a.first > b.first || a.second < b.second;
}

bool greater_predicted_affinity(Result &a, Result &b) {
	return a.second > b.second || a.first < b.first;
}

void Recommender::evaluate() {
	g_output_file << "Evaluating " << description() << " metrics..." << std::endl;
	clock_t start_clock = clock();
	Statistic mae = 0.0, rmse = 0.0;
	Ranked_Results ranked_results;
	Count num_data = 0;
	for (Data_Sparse_Matrix::const_iterator m_it = g_user_sr_affinities.begin(); m_it != g_user_sr_affinities.end(); ++m_it) {
		ID active_user_id = (*m_it).first;
		const Data_Sparse_Vector &sr_affinities = (*m_it).second;
		for (Data_Sparse_Vector::const_iterator v_it = sr_affinities.begin(); v_it != sr_affinities.end(); ++v_it) {
			ID sample_id = std::get<SAMPLE>((*v_it).second);
			if (_test_sample_id < g_num_samples && sample_id != _test_sample_id) {
				continue;
			}
			ID active_sr_id = (*v_it).first;
			Affinity actual_affinity = std::get<AFFINITY>((*v_it).second);
			Affinity predicted_affinity = predict_affinity(active_user_id, active_sr_id, actual_affinity);
			Affinity_Diff residual = predicted_affinity - actual_affinity;
			mae += abs(residual);
			rmse += pow((Statistic)residual, 2.0);
			ranked_results.push_back(std::make_pair(actual_affinity, predicted_affinity));
			num_data++;
		}
	}
	mae = num_data ? mae / num_data : 0.0;
	rmse = num_data ? sqrt(rmse / num_data) : 0.0;
	Statistic max_dcg = 0.0;
	std::sort(ranked_results.begin(), ranked_results.end(), greater_actual_affinity);
	for (size_t i = 0; i < ranked_results.size(); i++) {
		max_dcg += (pow(2.0, (Statistic)ranked_results[i].first) - 1.0) / log(i + 2.0);
	}
	std::random_shuffle(ranked_results.begin(), ranked_results.end());
	std::sort(ranked_results.begin(), ranked_results.end(), greater_predicted_affinity);
	Statistic ndcg = 0.0;
	for (size_t i = 0; i < ranked_results.size(); i++) {
		ndcg += (pow(2.0, (Statistic)ranked_results[i].first) - 1.0) / log(i + 2.0);
	}
	ndcg = ndcg / max_dcg;
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done evaluating " << description() << " metrics in " << seconds << " seconds" << std::endl;
	g_output_file << name() << " MAE: " << mae << std::endl;
	g_output_file << name() << " RMSE: " << rmse << std::endl;
	g_output_file << name() << " nDCG: " << ndcg << std::endl;
	g_output_file << std::endl;
	g_results_file << name() << "\t" << _test_sample_id << "\t" << mae << "\t" << rmse << "\t" << ndcg << std::endl;
}

Affinity Affinity_Average_Recommender::predict_affinity(ID, ID, Affinity) const {
	return g_average_affinity;
}

Affinity User_Average_Recommender::predict_affinity(ID active_user_id, ID, Affinity) const {
	return g_user_average_affinities.at(active_user_id);
}

Affinity Subreddit_Average_Recommender::predict_affinity(ID, ID active_sr_id, Affinity) const {
	return g_sr_average_affinities.at(active_sr_id);
}

std::string Combined_Average_Recommender::name() const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << "Combined-average (user fraction = " << _user_fraction << ")";
	return ss.str();
}

std::string Combined_Average_Recommender::description() const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << "combined average affinities (user fraction = " << _user_fraction << ")";
	return ss.str();
}

Affinity Combined_Average_Recommender::predict_affinity(ID active_user_id, ID active_sr_id, Affinity) const {
	return (Affinity)(g_user_average_affinities.at(active_user_id) * _user_fraction + g_sr_average_affinities.at(active_sr_id) * (1.0 - _user_fraction));
}

// Self-damping: http://grouplens.org/similarity-functions-for-user-user-collaborative-filtering/
User_Based_CF_Recommender::User_Based_CF_Recommender(Statistic rho, bool damping, ID test_sample_id) : Recommender(test_sample_id), _rho(rho), _damping(damping), _user_user_correlations(g_num_users * g_num_users, 0.0) {
	g_output_file << "Calculating user-user Pearson correlation coefficients (rho = " << _rho << (_damping ? ", with self-damping" : "") << ")..." << std::endl;
	clock_t start_clock = clock();
	for (ID user_a_id = 0; user_a_id < g_num_users; user_a_id++) {
		Affinity user_a_average_affinity = g_user_average_affinities.at(user_a_id);
		const Data_Sparse_Vector &user_a_affinities = g_user_sr_affinities.at(user_a_id);
		for (ID user_b_id = 0; user_b_id <= user_a_id; user_b_id++) {
			if (user_a_id == user_b_id) {
				Statistic correlation = 1.0;
				_user_user_correlations[user_a_id * g_num_users + user_b_id] = correlation;
				continue;
			}
			Affinity user_b_average_affinity = g_user_average_affinities.at(user_b_id);
			const Data_Sparse_Vector &user_b_affinities = g_user_sr_affinities.at(user_b_id);
			Statistic user_covariance = 0.0, user_a_variance = 0.0, user_b_variance = 0.0;
			for (Data_Sparse_Vector::const_iterator v_a_it = user_a_affinities.begin(); v_a_it != user_a_affinities.end(); ++v_a_it) {
				ID sr_id = (*v_a_it).first;
				ID sample_id = std::get<SAMPLE>((*v_a_it).second);
				if (sample_id == test_sample_id) {
					continue;
				}
				Affinity user_a_sr_affinity = std::get<AFFINITY>((*v_a_it).second);
				Affinity_Diff user_a_sr_norm_affinity = user_a_sr_affinity - user_a_average_affinity;
				Data_Sparse_Vector::const_iterator v_b_it = user_b_affinities.find(sr_id);
				if (v_b_it != user_b_affinities.end()) {
					Affinity user_b_sr_affinity = std::get<AFFINITY>((*v_b_it).second);
					Affinity_Diff user_b_sr_norm_affinity = user_b_sr_affinity - user_b_average_affinity;
					user_covariance += (Statistic)user_a_sr_norm_affinity * (Statistic)user_b_sr_norm_affinity;
					user_a_variance += pow((Statistic)user_a_sr_norm_affinity, 2.0);
					user_b_variance += pow((Statistic)user_b_sr_norm_affinity, 2.0);
				}
				else if (damping) {
					user_a_variance += pow((Statistic)user_a_sr_norm_affinity, 2.0);
				}
			}
			if (damping) {
				for (Data_Sparse_Vector::const_iterator v_b_it = user_b_affinities.begin(); v_b_it != user_b_affinities.end(); ++v_b_it) {
					ID sr_id = (*v_b_it).first;
					ID sample_id = std::get<SAMPLE>((*v_b_it).second);
					if (sample_id == test_sample_id) {
						continue;
					}
					Affinity user_b_sr_affinity = std::get<AFFINITY>((*v_b_it).second);
					Affinity_Diff user_b_sr_norm_affinity = user_b_sr_affinity - user_b_average_affinity;
					Data_Sparse_Vector::const_iterator v_a_it = user_a_affinities.find(sr_id);
					if (v_a_it == user_a_affinities.end()) {
						user_b_variance += pow((Statistic)user_b_sr_norm_affinity, 2.0);
					}
				}
			}
			Statistic correlation = user_a_variance && user_b_variance ? user_covariance / (sqrt(user_a_variance) * sqrt(user_b_variance)) : 0.0;
			if (rho != 1.0) { correlation *= pow(abs(correlation), rho - 1.0); }
			_user_user_correlations[user_a_id * g_num_users + user_b_id] = _user_user_correlations[user_b_id * g_num_users + user_a_id] = correlation;
		}
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done calculating user-user correlations in " << seconds << " seconds" << std::endl;
}

std::string User_Based_CF_Recommender::name() const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << "User-based CF (rho = " << _rho << (_damping ? ", with self-damping" : "") << ")";
	return ss.str();
}

std::string User_Based_CF_Recommender::description(void) const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << "user-based collaborative filtering (rho = " << _rho << (_damping ? ", with self-damping" : "") << ")";
	return ss.str();
}

Affinity User_Based_CF_Recommender::predict_affinity(ID active_user_id, ID active_sr_id, Affinity) const {
	Affinity active_user_average_affinity = g_user_average_affinities.at(active_user_id);
	Count active_user_row = active_user_id * g_num_users;
	Statistic predicted_numerator = 0.0, predicted_denominator = 0.0;
	const Data_Sparse_Vector &sr_affinities = g_sr_user_affinities.at(active_sr_id);
	for (ID other_user_id = 0; other_user_id < g_num_users; other_user_id++) {
		Data_Sparse_Vector::const_iterator v_o_it = sr_affinities.find(other_user_id);
		if (v_o_it == sr_affinities.end()) { continue; }
		Affinity other_user_sr_affinity = std::get<AFFINITY>((*v_o_it).second);
		Affinity other_user_average_affinity = g_user_average_affinities.at(other_user_id);
		Statistic correlation = _user_user_correlations.at(active_user_row + other_user_id);
		predicted_numerator += correlation * (other_user_sr_affinity - other_user_average_affinity);
		predicted_denominator += abs(correlation);
	}
	Statistic predicted_delta = predicted_denominator ? predicted_numerator / predicted_denominator : 0.0;
	Affinity predicted_affinity = active_user_average_affinity + (Affinity)predicted_delta;
	return predicted_affinity;
}

Subreddit_Based_CF_Recommender::Subreddit_Based_CF_Recommender(Statistic rho, bool damping, ID test_sample_id) : Recommender(test_sample_id), _rho(rho), _damping(damping), _sr_sr_correlations(g_num_srs * g_num_srs, 0.0) {
	g_output_file << "Calculating subreddit-subreddit Pearson correlation coefficients (rho = " << rho << (damping ? ", with self-damping" : "") << ")..." << std::endl;
	clock_t start_clock = clock();
	for (ID sr_a_id = 0; sr_a_id < g_num_srs; sr_a_id++) {
		Affinity sr_a_average_affinity = g_sr_average_affinities.at(sr_a_id);
		const Data_Sparse_Vector &sr_a_affinities = g_sr_user_affinities.at(sr_a_id);
		for (ID sr_b_id = 0; sr_b_id <= sr_a_id; sr_b_id++) {
			if (sr_a_id == sr_b_id) {
				Statistic correlation = 1.0;
				_sr_sr_correlations[sr_a_id * g_num_srs + sr_b_id] = correlation;
				continue;
			}
			Affinity sr_b_average_affinity = g_sr_average_affinities.at(sr_b_id);
			const Data_Sparse_Vector &sr_b_affinities = g_sr_user_affinities.at(sr_b_id);
			Statistic sr_covariance = 0.0, sr_a_variance = 0.0, sr_b_variance = 0.0;
			for (Data_Sparse_Vector::const_iterator v_a_it = sr_a_affinities.begin(); v_a_it != sr_a_affinities.end(); ++v_a_it) {
				ID user_id = (*v_a_it).first;
				ID sample_id = std::get<SAMPLE>((*v_a_it).second);
				if (sample_id == test_sample_id) {
					continue;
				}
				Affinity sr_a_user_affinity = std::get<AFFINITY>((*v_a_it).second);
				Affinity_Diff sr_a_user_norm_affinity = sr_a_user_affinity - sr_a_average_affinity;
				Data_Sparse_Vector::const_iterator v_b_it = sr_b_affinities.find(user_id);
				if (v_b_it != sr_b_affinities.end()) {
					Affinity sr_b_user_affinity = std::get<AFFINITY>((*v_b_it).second);
					Affinity_Diff sr_b_user_norm_affinity = sr_b_user_affinity - sr_b_average_affinity;
					sr_covariance += (Statistic)sr_a_user_norm_affinity * (Statistic)sr_b_user_norm_affinity;
					sr_a_variance += pow((Statistic)sr_a_user_norm_affinity, 2.0);
					sr_b_variance += pow((Statistic)sr_b_user_norm_affinity, 2.0);
				}
				else if (damping) {
					sr_a_variance += pow((Statistic)sr_a_user_norm_affinity, 2.0);
				}
			}
			if (damping) {
				for (Data_Sparse_Vector::const_iterator v_b_it = sr_b_affinities.begin(); v_b_it != sr_b_affinities.end(); ++v_b_it) {
					ID user_id = (*v_b_it).first;
					ID sample_id = std::get<SAMPLE>((*v_b_it).second);
					if (sample_id == test_sample_id) {
						continue;
					}
					Affinity sr_b_user_affinity = std::get<AFFINITY>((*v_b_it).second);
					Affinity_Diff sr_b_user_norm_affinity = sr_b_user_affinity - sr_b_average_affinity;
					Data_Sparse_Vector::const_iterator v_a_it = sr_a_affinities.find(user_id);
					if (v_a_it == sr_a_affinities.end()) {
						sr_b_variance += pow((Statistic)sr_b_user_norm_affinity, 2.0);
					}
				}
			}
			Statistic correlation = sr_a_variance && sr_b_variance ? sr_covariance / (sqrt(sr_a_variance) * sqrt(sr_b_variance)) : 0.0;
			if (rho != 1.0) { correlation *= pow(abs(correlation), rho - 1.0); }
			_sr_sr_correlations[sr_a_id * g_num_srs + sr_b_id] = _sr_sr_correlations[sr_b_id * g_num_srs + sr_a_id] = correlation;
		}
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done calculating subreddit-subreddit correlations in " << seconds << " seconds" << std::endl;
}

std::string Subreddit_Based_CF_Recommender::name() const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << "Subreddit-based CF (rho = " << _rho << (_damping ? ", with self-damping" : "") << ")";
	return ss.str();
}

std::string Subreddit_Based_CF_Recommender::description(void) const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << "subreddit-based collaborative filtering (rho = " << _rho << (_damping ? ", with self-damping" : "") << ")";
	return ss.str();
}

Affinity Subreddit_Based_CF_Recommender::predict_affinity(ID active_user_id, ID active_sr_id, Affinity) const {
	Affinity active_sr_average_affinity = g_sr_average_affinities.at(active_sr_id);
	Count active_sr_row = active_sr_id * g_num_srs;
	Statistic predicted_numerator = 0.0, predicted_denominator = 0.0;
	const Data_Sparse_Vector &user_affinities = g_user_sr_affinities.at(active_user_id);
	for (ID other_sr_id = 0; other_sr_id < g_num_srs; other_sr_id++) {
		Data_Sparse_Vector::const_iterator v_o_it = user_affinities.find(other_sr_id);
		if (v_o_it == user_affinities.end()) { continue; }
		Affinity other_sr_user_affinity = std::get<AFFINITY>((*v_o_it).second);
		Affinity other_sr_average_affinity = g_sr_average_affinities.at(other_sr_id);
		Statistic correlation = _sr_sr_correlations.at(active_sr_row + other_sr_id);
		predicted_numerator += correlation * (other_sr_user_affinity - other_sr_average_affinity);
		predicted_denominator += abs(correlation);
	}
	Statistic predicted_delta = predicted_denominator ? predicted_numerator / predicted_denominator : 0.0;
	Affinity predicted_affinity = active_sr_average_affinity + (Affinity)predicted_delta;
	return predicted_affinity;
}

SVD_Recommender::SVD_Recommender(ID test_sample_id) : Recommender(test_sample_id), _k(0), _pure(false) {
	std::string svd_py_filepath("collaborative/svd.py");
	g_output_file << "Running " << svd_py_filepath << "..." << std::endl;
	clock_t start_clock = clock();
	std::ostringstream ss;
	ss << g_directory << svd_py_filepath << " " << g_num_users << " " << g_num_srs << " " << g_max_num_singular_values << " " << test_sample_id;
	int return_code = system(ss.str().c_str());
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done running " << svd_py_filepath << " in " << seconds << " seconds" << std::endl;
	g_output_file << svd_py_filepath << " returned " << return_code << std::endl;
	g_output_file << std::endl;

	g_output_file << "Reading SVD matrix files..." << std::endl;
	start_clock = clock();

	std::string svd_matrix_vt_filename("svd_matrix_vt_st.dat");
	std::ifstream svd_matrix_vt_file(g_directory + svd_matrix_vt_filename);
	g_output_file << "Reading " << svd_matrix_vt_filename << "..." << std::endl;
	size_t num_rows, num_cols;
	svd_matrix_vt_file >> num_rows >> num_cols;
	_svd_matrix_vt.resize(num_rows);
	for (size_t i = 0; i < num_rows; i++) {
		_svd_matrix_vt[i].resize(num_cols);
		for (size_t j = 0; j < num_cols; j++) {
			svd_matrix_vt_file >> _svd_matrix_vt[i][j];
		}
	}
	g_output_file << "Done reading " << svd_matrix_vt_filename << std::endl;

	std::string svd_matrix_usqrtsigma_filename("svd_matrix_usqrtsigma_st.dat");
	std::ifstream svd_matrix_usqrtsigma_file(g_directory + svd_matrix_usqrtsigma_filename);
	g_output_file << "Reading " << svd_matrix_usqrtsigma_filename << "..." << std::endl;
	svd_matrix_usqrtsigma_file >> num_rows >> num_cols;
	_svd_matrix_usqrtsigma.resize(num_rows);
	for (size_t i = 0; i < num_rows; i++) {
		_svd_matrix_usqrtsigma[i].resize(num_cols);
		for (size_t j = 0; j < num_cols; j++) {
			svd_matrix_usqrtsigma_file >> _svd_matrix_usqrtsigma[i][j];
		}
	}
	g_output_file << "Done reading " << svd_matrix_usqrtsigma_filename << std::endl;

	std::string svd_matrix_sqrtsigmavt_filename("svd_matrix_sqrtsigmavt_st.dat");
	std::ifstream svd_matrix_sqrtsigmavt_file(g_directory + svd_matrix_sqrtsigmavt_filename);
	g_output_file << "Reading " << svd_matrix_sqrtsigmavt_filename << "..." << std::endl;
	svd_matrix_sqrtsigmavt_file >> num_rows >> num_cols;
	_svd_matrix_sqrtsigmavt.resize(num_rows);
	for (size_t i = 0; i < num_rows; i++) {
		_svd_matrix_sqrtsigmavt[i].resize(num_cols);
		for (size_t j = 0; j < num_cols; j++) {
			svd_matrix_sqrtsigmavt_file >> _svd_matrix_sqrtsigmavt[i][j];
		}
	}
	g_output_file << "Done reading " << svd_matrix_sqrtsigmavt_filename << std::endl;

	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done reading SVD matrix files in " << seconds << " seconds" << std::endl;
	g_output_file << std::endl;
}

std::string SVD_Recommender::name() const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << (_pure ? "PureSVD" : "OrigSVD") << " (k = " << _k << ")";
	return ss.str();
}

std::string SVD_Recommender::description(void) const {
	std::ostringstream ss;
	ss.precision(g_precision);
	ss << (_pure ? "PureSVD" : "OrigSVD") << " (k = " << _k << ")";
	return ss.str();
}

Affinity SVD_Recommender::predict_affinity(ID active_user_id, ID active_sr_id, Affinity) const {
	Affinity predicted_affinity;
	if (_pure) {
		const Data_Sparse_Vector &sr_affinities = g_user_sr_affinities.at(active_user_id);
		Statistic predicted_norm_affinity = 0.0;
		for (Count i = 0; i < _k; i++) {
			Count j = g_max_num_singular_values - i - 1;
			Statistic component = 0.0;
			const Statistic_Vector &right_singular_vector = _svd_matrix_vt[j];
			for (Data_Sparse_Vector::const_iterator v_it = sr_affinities.begin(); v_it != sr_affinities.end(); ++v_it) {
				ID nonzero_sr_id = (*v_it).first;
				Affinity nonzero_sr_affinity = std::get<AFFINITY>((*v_it).second);
				component += nonzero_sr_affinity * right_singular_vector[nonzero_sr_id];
			}
			predicted_norm_affinity += component * right_singular_vector[active_sr_id];
		}
		predicted_affinity = (Affinity)predicted_norm_affinity + g_average_affinity;
	}
	else {
		Statistic predicted_norm_affinity = 0.0;
		for (Count i = 0; i < _k; i++) {
			Count j = g_max_num_singular_values - i - 1;
			Statistic component = _svd_matrix_usqrtsigma[active_user_id][j] * _svd_matrix_sqrtsigmavt[j][active_sr_id];
			predicted_norm_affinity += component;
		}
		predicted_affinity = (Affinity)predicted_norm_affinity + g_average_affinity;
	}
	return predicted_affinity;
}

int main() {
	std::srand(g_random_seed);
	g_output_file.precision(g_precision);
	g_results_file.precision(g_precision);
	clock_t total_start_clock = clock();
	read_affinities_file();
	g_results_file << "recommender_name\ttest_sample_id\tmae\trmse\tndcg" << std::endl;
	for (ID test_sample_id = 0; test_sample_id <= g_num_samples; test_sample_id++) {
		g_output_file << "================================================================================" << std::endl;
		g_output_file << "Testing systems with ";
		if (test_sample_id == g_num_samples) {
			g_output_file << "all data samples";
		}
		else {
			g_output_file << "data sample #" << test_sample_id;
		}
		g_output_file << std::endl;
		g_output_file << "================================================================================" << std::endl;
		g_output_file << std::endl;
		calc_average_user_affinities(test_sample_id);
		calc_average_sr_affinities(test_sample_id);
		{ Worst_Recommender(test_sample_id).evaluate(); }
		{ Random_Recommender(test_sample_id).evaluate(); }
		{ Affinity_Average_Recommender(test_sample_id).evaluate(); }
		{ User_Average_Recommender(test_sample_id).evaluate(); }
		{ Subreddit_Average_Recommender(test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.1, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.2, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.3, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.4, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.5, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.6, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.7, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.8, test_sample_id).evaluate(); }
		{ Combined_Average_Recommender(0.9, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(1.0, false, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(1.5, false, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(2.0, false, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(2.5, false, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(3.0, false, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(3.5, false, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(4.0, false, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(1.0, true, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(1.5, true, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(2.0, true, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(2.5, true, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(3.0, true, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(3.5, true, test_sample_id).evaluate(); }
		{ User_Based_CF_Recommender(4.0, true, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(1.0, false, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(1.5, false, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(2.0, false, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(2.5, false, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(3.0, false, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(3.5, false, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(4.0, false, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(1.0, true, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(1.5, true, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(2.0, true, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(2.5, true, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(3.0, true, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(3.5, true, test_sample_id).evaluate(); }
		{ Subreddit_Based_CF_Recommender(4.0, true, test_sample_id).evaluate(); }
		{
			SVD_Recommender svd_recommender(test_sample_id);
			for (Count k = 0; k <= 20; k++) {
				svd_recommender.setup(k, false);
				svd_recommender.evaluate();
			}
			for (Count k = 50; k <= 500; k += 50) {
				svd_recommender.setup(k, false);
				svd_recommender.evaluate();
			}
			for (Count k = 0; k <= 20; k++) {
				svd_recommender.setup(k, true);
				svd_recommender.evaluate();
			}
			for (Count k = 50; k <= 500; k += 50) {
				svd_recommender.setup(k, true);
				svd_recommender.evaluate();
			}
		}
	}
	clock_t total_end_clock = clock();
	Statistic total_seconds = seconds_between(total_start_clock, total_end_clock);
	g_output_file << "Total elapsed time: " << total_seconds << " seconds" << std::endl;
	return 0;
}
