#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>
#include <algorithm>
#include <limits>
#include <ctime>

typedef double Statistic;
typedef unsigned int Count;
typedef unsigned int ID;
typedef int Vote;
typedef std::string Hash;
typedef std::unordered_map<Hash, ID> Hash_IDs;
typedef std::unordered_map<ID, Hash> ID_Hashes;
typedef std::tuple<Count, Count, Count> Vote_Counts;
typedef std::unordered_map<ID, Count> Counts;
typedef std::unordered_map<ID, Vote_Counts> Votes_Sparse_Vector;
typedef std::unordered_map<ID, Votes_Sparse_Vector> Votes_Sparse_Matrix;
typedef std::map<Count, Count> Count_Counts;
typedef std::map<Count, Vote_Counts> Vote_Count_Counts;

enum { UP = 0, DOWN = 1, TOTAL = 2 };

std::string g_directory("C:/Users/Remy/Desktop/590/");
std::string g_output_filename("affinities_output.log");
std::ofstream g_output_file(g_directory + g_output_filename);
size_t g_precision = 12;
unsigned int g_random_seed = 42;

ID_Hashes g_user_id_hashes, g_post_id_hashes, g_sr_id_hashes;
Counts g_user_post_counts,  g_sr_post_counts;
Votes_Sparse_Matrix g_user_sr_votes, g_sr_user_votes;
Count g_num_votes = 0;
Count g_num_samples = 10;

Statistic seconds_between(clock_t start, clock_t end) {
	return (Statistic)(end - start) / CLOCKS_PER_SEC;
}

void read_votes_file() {
	Hash_IDs user_hash_ids, post_hash_ids, sr_hash_ids;
	std::string votes_filename("publicvotes-20101018_votes.dump");
	std::ifstream votes_file(g_directory + votes_filename);
	g_output_file << "Reading " << votes_filename << "..." << std::endl;
	clock_t start_clock = clock();
	Hash user_hash, post_hash, sr_hash;
	Vote vote;
	while (votes_file >> user_hash >> post_hash >> sr_hash >> vote) {
		if (user_hash_ids.find(user_hash) == user_hash_ids.end()) { user_hash_ids[user_hash] = (ID)user_hash_ids.size(); }
		if (post_hash_ids.find(post_hash) == post_hash_ids.end()) { post_hash_ids[post_hash] = (ID)post_hash_ids.size(); }
		if (sr_hash_ids.find(sr_hash) == sr_hash_ids.end()) { sr_hash_ids[sr_hash] = (ID)sr_hash_ids.size(); }
		ID user_id = user_hash_ids[user_hash], post_id = post_hash_ids[post_hash], sr_id = sr_hash_ids[sr_hash];
		g_user_id_hashes[user_id] = user_hash;
		g_post_id_hashes[post_id] = post_hash;
		g_sr_id_hashes[sr_id] = sr_hash;
		g_num_votes++;
		g_user_post_counts[user_id]++;
		g_sr_post_counts[sr_id]++;
		if (vote > 0) {
			std::get<UP>(g_user_sr_votes[user_id][sr_id])++;
			std::get<UP>(g_sr_user_votes[sr_id][user_id])++;
		}
		else {
			std::get<DOWN>(g_user_sr_votes[user_id][sr_id])++;
			std::get<DOWN>(g_sr_user_votes[sr_id][user_id])++;
		}
		std::get<TOTAL>(g_user_sr_votes[user_id][sr_id])++;
		std::get<TOTAL>(g_sr_user_votes[sr_id][user_id])++;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done reading " << votes_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << g_num_votes << " votes, on " << g_post_id_hashes.size() << " posts, by " << g_user_id_hashes.size() << " users, in " << g_sr_id_hashes.size() << " subreddits" << std::endl;
	g_output_file << std::endl;
}

void write_id_hash_maps() {
	g_output_file << "Writing ID-to-hash maps..." << std::endl;
	clock_t start_clock = clock();

	std::string user_ids_to_hashes_filename("user_id_hashes.tsv");
	std::ofstream user_ids_to_hashes_file(g_directory + user_ids_to_hashes_filename);
	user_ids_to_hashes_file.precision(g_precision);
	g_output_file << "Writing " << user_ids_to_hashes_filename << "..." << std::endl;
	user_ids_to_hashes_file << "user_id\tuser_hash" << std::endl;
	for (ID user_id = 0; user_id < g_user_id_hashes.size(); user_id++) {
		Hash user_hash = g_user_id_hashes[user_id];
		user_ids_to_hashes_file << user_id << "\t" << user_hash << std::endl;
	}
	g_output_file << "Done writing " << user_ids_to_hashes_filename << std::endl;

	std::string post_ids_to_hashes_filename("post_id_hashes.tsv");
	std::ofstream post_ids_to_hashes_file(g_directory + post_ids_to_hashes_filename);
	post_ids_to_hashes_file.precision(g_precision);
	g_output_file << "Writing " << post_ids_to_hashes_filename << "..." << std::endl;
	post_ids_to_hashes_file << "user_id\tuser_hash" << std::endl;
	for (ID post_id = 0; post_id < g_user_id_hashes.size(); post_id++) {
		Hash post_hash = g_post_id_hashes[post_id];
		post_ids_to_hashes_file << post_id << "\t" << post_hash << std::endl;
	}
	g_output_file << "Done writing " << post_ids_to_hashes_filename << std::endl;

	std::string sr_ids_to_hashes_filename("sr_id_hashes.tsv");
	std::ofstream sr_ids_to_hashes_file(g_directory + sr_ids_to_hashes_filename);
	sr_ids_to_hashes_file.precision(g_precision);
	g_output_file << "Writing " << sr_ids_to_hashes_filename << "..." << std::endl;
	sr_ids_to_hashes_file << "sr_id\tsr_hash" << std::endl;
	for (ID sr_id = 0; sr_id < g_sr_id_hashes.size(); sr_id++) {
		Hash sr_hash = g_sr_id_hashes[sr_id];
		sr_ids_to_hashes_file << sr_id << "\t" << sr_hash << std::endl;
	}
	g_output_file << "Done writing " << sr_ids_to_hashes_filename << std::endl;

	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing ID-to-hash maps in " << seconds << " seconds" << std::endl;
	g_output_file << std::endl;
}

void calc_user_sr_affinities() {
	Count num_affinities = 0;
	Statistic avg_upvote_proportion = 0.0, avg_vote_total = 0.0;
	Count median_vote_total = 0;

	std::vector<Count> vote_totals;
	g_output_file << "Calculating user-subreddit upvote proportions..." << std::endl;
	clock_t start_clock = clock();
	vote_totals.reserve(g_num_votes);
	for (Votes_Sparse_Matrix::const_iterator m_it = g_user_sr_votes.begin(); m_it != g_user_sr_votes.end(); ++m_it) {
		const Votes_Sparse_Vector &sr_votes = (*m_it).second;
		for (Votes_Sparse_Vector::const_iterator v_it = sr_votes.begin(); v_it != sr_votes.end(); ++v_it) {
			Vote_Counts vote_counts = (*v_it).second;
			Count n_upvotes, n_downvotes, n_votes;
			std::tie(n_upvotes, n_downvotes, n_votes) = vote_counts;
			num_affinities++;
			Statistic upvote_proportion = (Statistic)n_upvotes / (Statistic)n_votes;
			avg_upvote_proportion += upvote_proportion;
			avg_vote_total += n_votes;
			vote_totals.push_back(n_votes);
		}
	}
	avg_upvote_proportion /= num_affinities;
	avg_vote_total /= num_affinities;
	size_t median_vote_total_index = vote_totals.size() / 2;
	std::nth_element(vote_totals.begin(), vote_totals.begin() + median_vote_total_index, vote_totals.end());
    median_vote_total = vote_totals[median_vote_total_index];
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done calculating user-subreddit upvote proportions in " << seconds << " seconds" << std::endl;
	g_output_file << "Average user-subreddit upvote proportion is: " << avg_upvote_proportion << std::endl;
	g_output_file << "Average total user-subreddit vote count is: " << avg_vote_total << std::endl;
	g_output_file << "Median total user-subreddit vote count is: " << median_vote_total << std::endl;
	g_output_file << std::endl;
	vote_totals.clear();

	std::vector<Count> indices(num_affinities);
	for (Count i = 0; i < num_affinities; i++) {
		indices[i] = i;
	}
	std::random_shuffle(indices.begin(), indices.end());

	std::string affinities_filename("affinities.tsv");
	std::ofstream affinities_file(g_directory + affinities_filename);
	g_output_file << "Calculating user-subreddit affinities; writing " << affinities_filename << "..." << std::endl;
	start_clock = clock();
	affinities_file.precision(g_precision);
	affinities_file << "user_id\tsr_id\taffinity\tn_votes\tsample_id" << std::endl;
	Statistic affinity_bias = median_vote_total * avg_upvote_proportion;
	Statistic avg_affinity = 0.0;
	Count index = 0;
	Count partition_size = num_affinities / g_num_samples;
	for (Votes_Sparse_Matrix::const_iterator m_it = g_user_sr_votes.begin(); m_it != g_user_sr_votes.end(); ++m_it) {
		ID user_id = (*m_it).first;
		const Votes_Sparse_Vector &sr_votes = (*m_it).second;
		for (Votes_Sparse_Vector::const_iterator v_it = sr_votes.begin(); v_it != sr_votes.end(); ++v_it) {
			ID sr_id = (*v_it).first;
			ID sample_id = (ID)(indices[index] / partition_size);
			if (sample_id >= g_num_samples) {
				sample_id = (ID)(g_num_samples - 1);
			}
			Vote_Counts vote_counts = (*v_it).second;
			Count n_upvotes, n_downvotes, n_votes;
			std::tie(n_upvotes, n_downvotes, n_votes) = vote_counts;
			Statistic affinity = (Statistic)(affinity_bias + n_upvotes) / (Statistic)(median_vote_total + n_votes);
			avg_affinity += affinity;
			affinities_file << user_id << "\t" << sr_id << "\t" << affinity << "\t" << n_votes << "\t" << sample_id << std::endl;
			index++;
		}
	}
	avg_affinity /= num_affinities;
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << affinities_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << num_affinities << " user-subreddit affinities, average " << avg_affinity << std::endl;
	g_output_file << std::endl;
}

void count_user_post_counts() {
	Count_Counts user_post_count_counts;

	g_output_file << "Counting users' post counts..." << std::endl;
	clock_t start_clock = clock();
	for (Counts::const_iterator u_it = g_user_post_counts.begin(); u_it != g_user_post_counts.end(); ++u_it) {
		Count user_post_count = (*u_it).second;
		user_post_count_counts[user_post_count]++;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done counting users' post counts in " << seconds << " seconds" << std::endl;

	std::string user_post_count_counts_filename("user_post_count_counts.tsv");
	std::ofstream user_post_count_counts_file(g_directory + user_post_count_counts_filename);
	g_output_file << "Writing " << user_post_count_counts_filename << "..." << std::endl;
	start_clock = clock();
	user_post_count_counts_file.precision(g_precision);
	user_post_count_counts_file << "user_post_count\tnum_users" << std::endl;
	Count min_user_post_count = user_post_count_counts.begin()->first;
	Count max_user_post_count = user_post_count_counts.rbegin()->first;
	Statistic avg_user_post_count = 0.0;
	Count median_user_post_count = 0, median_user_post_count_counter = 0;
	for (Count user_post_count = 0; user_post_count <= max_user_post_count; user_post_count++) {
		Count n_users = user_post_count_counts[user_post_count];
		if (n_users > 0) {
			user_post_count_counts_file << user_post_count << "\t" << n_users << std::endl;
		}
		avg_user_post_count += ((Statistic)user_post_count * (Statistic)n_users) / (Statistic)g_user_post_counts.size();
		median_user_post_count_counter += n_users;
		if (median_user_post_count == 0 && median_user_post_count_counter >= g_user_post_counts.size() / 2) {
			median_user_post_count = user_post_count;
		}
	}
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << user_post_count_counts_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << "Each user voted on " << min_user_post_count << " to " << max_user_post_count << " posts, average " << avg_user_post_count << ", median " << median_user_post_count << std::endl;
	for (Counts::const_iterator u_it = g_user_post_counts.begin(); u_it != g_user_post_counts.end(); ++u_it) {
		ID user_id = (*u_it).first;
		Count user_post_count = (*u_it).second;
		if (user_post_count == max_user_post_count) {
			g_output_file << "A user who voted on the maximum " << max_user_post_count << " posts is: #" << user_id << " (" << g_user_id_hashes[user_id] << ")" << std::endl;
			break;
		}
	}
	g_output_file << std::endl;
}

void count_sr_post_counts() {
	Count_Counts sr_post_count_counts;

	g_output_file << "Counting subreddits' post counts..." << std::endl;
	clock_t start_clock = clock();
	for (Counts::const_iterator s_it = g_sr_post_counts.begin(); s_it != g_sr_post_counts.end(); ++s_it) {
		Count sr_post_count = (*s_it).second;
		sr_post_count_counts[sr_post_count]++;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done counting subreddits' post counts in " << seconds << " seconds" << std::endl;

	std::string sr_post_count_counts_filename("sr_post_count_counts.tsv");
	std::ofstream sr_post_count_counts_file(g_directory + sr_post_count_counts_filename);
	g_output_file << "Writing " << sr_post_count_counts_filename << "..." << std::endl;
	start_clock = clock();
	sr_post_count_counts_file.precision(g_precision);
	sr_post_count_counts_file << "sr_post_count\tnum_srs" << std::endl;
	Count min_sr_post_count = sr_post_count_counts.begin()->first;
	Count max_sr_post_count = sr_post_count_counts.rbegin()->first;
	Statistic avg_sr_post_count = 0.0;
	Count median_sr_post_count = 0, median_sr_post_count_counter = 0;
	for (Count sr_post_count = 0; sr_post_count <= max_sr_post_count; sr_post_count++) {
		Count n_srs = sr_post_count_counts[sr_post_count];
		if (n_srs > 0) {
			sr_post_count_counts_file << sr_post_count << "\t" << n_srs << std::endl;
		}
		avg_sr_post_count += ((Statistic)sr_post_count * (Statistic)n_srs) / (Statistic)g_sr_post_counts.size();
		median_sr_post_count_counter += n_srs;
		if (median_sr_post_count == 0 && median_sr_post_count_counter >= g_sr_post_counts.size() / 2) {
			median_sr_post_count = sr_post_count;
		}
	}
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << sr_post_count_counts_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << "Each subreddit has " << min_sr_post_count << " to " << max_sr_post_count << " posts, average " << avg_sr_post_count << ", median " << median_sr_post_count << std::endl;
	for (Counts::const_iterator s_it = g_sr_post_counts.begin(); s_it != g_sr_post_counts.end(); ++s_it) {
		ID sr_id = (*s_it).first;
		Count sr_post_count = (*s_it).second;
		if (sr_post_count == max_sr_post_count) {
			g_output_file << "A subreddit with the maximum " << max_sr_post_count << " posts is: #" << sr_id << " (" << g_sr_id_hashes[sr_id] << ")" << std::endl;
			break;
		}
	}
	g_output_file << std::endl;
}

void count_user_sr_counts() {
	Count_Counts user_sr_count_counts;

	g_output_file << "Counting users' subreddit counts..." << std::endl;
	clock_t start_clock = clock();
	for (Votes_Sparse_Matrix::const_iterator m_it = g_user_sr_votes.begin(); m_it != g_user_sr_votes.end(); ++m_it) {
		const Votes_Sparse_Vector &sr_votes = (*m_it).second;
		Count user_sr_count = (Count)sr_votes.size();
		user_sr_count_counts[user_sr_count]++;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done counting users' subreddit counts in " << seconds << " seconds" << std::endl;

	std::string user_sr_count_counts_filename("user_sr_count_counts.tsv");
	std::ofstream user_sr_count_counts_file(g_directory + user_sr_count_counts_filename);
	g_output_file << "Writing " << user_sr_count_counts_filename << "..." << std::endl;
	start_clock = clock();
	user_sr_count_counts_file.precision(g_precision);
	user_sr_count_counts_file << "user_sr_count\tnum_users" << std::endl;
	Count min_user_sr_count = user_sr_count_counts.begin()->first;
	Count max_user_sr_count = user_sr_count_counts.rbegin()->first;
	Statistic avg_user_sr_count = 0.0;
	Count median_user_sr_count = 0, median_user_sr_count_counter = 0;
	for (Count user_sr_count = 0; user_sr_count <= max_user_sr_count; user_sr_count++) {
		Count n_users = user_sr_count_counts[user_sr_count];
		if (n_users > 0) {
			user_sr_count_counts_file << user_sr_count << "\t" << n_users << std::endl;
		}
		avg_user_sr_count += ((Statistic)user_sr_count * (Statistic)n_users) / (Statistic)g_user_post_counts.size();
		median_user_sr_count_counter += n_users;
		if (median_user_sr_count == 0 && median_user_sr_count_counter >= g_user_post_counts.size() / 2) {
			median_user_sr_count = user_sr_count;
		}
	}
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << user_sr_count_counts_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << "Each user has voted on posts in " << min_user_sr_count << " to " << max_user_sr_count << " subreddits, average " << avg_user_sr_count << ", median " << median_user_sr_count << std::endl;
	for (Votes_Sparse_Matrix::const_iterator m_it = g_user_sr_votes.begin(); m_it != g_user_sr_votes.end(); ++m_it) {
		ID user_id = (*m_it).first;
		const Votes_Sparse_Vector &sr_votes = (*m_it).second;
		Count user_sr_count = (Count)sr_votes.size();
		if (user_sr_count == max_user_sr_count) {
			g_output_file << "A user with votes on posts in the maximum " << max_user_sr_count << " subreddits is: #" << user_id << " (" << g_user_id_hashes[user_id] << ")" << std::endl;
			break;
		}
	}
	g_output_file << std::endl;
}

void count_sr_user_counts() {
	Count_Counts sr_user_count_counts;

	g_output_file << "Counting subreddits' user counts..." << std::endl;
	clock_t start_clock = clock();
	for (Votes_Sparse_Matrix::const_iterator m_it = g_sr_user_votes.begin(); m_it != g_sr_user_votes.end(); ++m_it) {
		const Votes_Sparse_Vector &user_votes = (*m_it).second;
		Count sr_user_count = (Count)user_votes.size();
		sr_user_count_counts[sr_user_count]++;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done counting subreddits' user counts in " << seconds << " seconds" << std::endl;

	std::string sr_user_count_counts_filename("sr_user_count_counts.tsv");
	std::ofstream sr_user_count_counts_file(g_directory + sr_user_count_counts_filename);
	g_output_file << "Writing " << sr_user_count_counts_filename << "..." << std::endl;
	start_clock = clock();
	sr_user_count_counts_file.precision(g_precision);
	sr_user_count_counts_file << "sr_user_count\tnum_srs" << std::endl;
	Count min_sr_user_count = sr_user_count_counts.begin()->first;
	Count max_sr_user_count = sr_user_count_counts.rbegin()->first;
	Statistic avg_sr_user_count = 0.0;
	Count median_sr_user_count = 0, median_sr_user_count_counter = 0;
	for (Count sr_user_count = 0; sr_user_count <= max_sr_user_count; sr_user_count++) {
		Count n_srs = sr_user_count_counts[sr_user_count];
		if (n_srs > 0) {
			sr_user_count_counts_file << sr_user_count << "\t" << n_srs << std::endl;
		}
		avg_sr_user_count += ((Statistic)sr_user_count * (Statistic)n_srs) / (Statistic)g_sr_post_counts.size();
		median_sr_user_count_counter += n_srs;
		if (median_sr_user_count == 0 && median_sr_user_count_counter >= g_sr_post_counts.size() / 2) {
			median_sr_user_count = sr_user_count;
		}
	}
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << sr_user_count_counts_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << "Each subreddit has posts with votes by " << min_sr_user_count << " to " << max_sr_user_count << " users, average " << avg_sr_user_count << ", median " << median_sr_user_count << std::endl;
	for (Votes_Sparse_Matrix::const_iterator m_it = g_sr_user_votes.begin(); m_it != g_sr_user_votes.end(); ++m_it) {
		ID sr_id = (*m_it).first;
		const Votes_Sparse_Vector &user_votes = (*m_it).second;
		Count sr_user_count = (Count)user_votes.size();
		if (sr_user_count == max_sr_user_count) {
			g_output_file << "A subreddit with posts with votes by the maximum " << max_sr_user_count << " users is: #" << sr_id << " (" << g_sr_id_hashes[sr_id] << ")" << std::endl;
			break;
		}
	}
	g_output_file << std::endl;
}

void count_user_vote_counts() {
	Vote_Count_Counts user_vote_count_counts;

	g_output_file << "Counting users' vote counts..." << std::endl;
	clock_t start_clock = clock();
	for (Votes_Sparse_Matrix::const_iterator m_it = g_user_sr_votes.begin(); m_it != g_user_sr_votes.end(); ++m_it) {
		const Votes_Sparse_Vector &sr_votes = (*m_it).second;
		Count user_upvote_count = 0, user_downvote_count = 0, user_vote_count = 0;
		for (Votes_Sparse_Vector::const_iterator v_it = sr_votes.begin(); v_it != sr_votes.end(); ++v_it) {
			Vote_Counts vote_counts = (*v_it).second;
			user_upvote_count += std::get<UP>(vote_counts);
			user_downvote_count += std::get<DOWN>(vote_counts);
			user_vote_count += std::get<TOTAL>(vote_counts);
		}
		std::get<UP>(user_vote_count_counts[user_upvote_count])++;
		std::get<DOWN>(user_vote_count_counts[user_downvote_count])++;
		std::get<TOTAL>(user_vote_count_counts[user_vote_count])++;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done counting users' vote counts in " << seconds << " seconds" << std::endl;

	std::string user_vote_count_counts_filename("user_vote_count_counts.tsv");
	std::ofstream user_vote_count_counts_file(g_directory + user_vote_count_counts_filename);
	g_output_file << "Writing " << user_vote_count_counts_filename << "..." << std::endl;
	start_clock = clock();
	user_vote_count_counts_file.precision(g_precision);
	user_vote_count_counts_file << "user_vote_count\tnum_up\tnum_down\tnum_total" << std::endl;
	Count min_user_vote_count = user_vote_count_counts.begin()->first;
	Count max_user_vote_count = user_vote_count_counts.rbegin()->first;
	Statistic avg_user_upvote_count = 0.0, avg_user_downvote_count = 0.0, avg_user_vote_count = 0.0;
	for (Count user_vote_count = 0; user_vote_count <= max_user_vote_count; user_vote_count++) {
		Vote_Counts ns_users = user_vote_count_counts[user_vote_count];
		Count n_up_users, n_down_users, n_users;
		std::tie(n_up_users, n_down_users, n_users) = ns_users;
		if (n_users > 0) {
			user_vote_count_counts_file << user_vote_count << "\t" << n_up_users << "\t" << n_down_users << "\t" << n_users << std::endl;
		}
		avg_user_upvote_count += ((Statistic)user_vote_count * (Statistic)n_up_users) / (Statistic)g_user_post_counts.size();
		avg_user_downvote_count += ((Statistic)user_vote_count * (Statistic)n_down_users) / (Statistic)g_user_post_counts.size();
		avg_user_vote_count += ((Statistic)user_vote_count * (Statistic)n_users) / (Statistic)g_user_post_counts.size();
	}
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << user_vote_count_counts_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << "Each user placed " << min_user_vote_count << " to " << max_user_vote_count << " votes, average " << avg_user_vote_count << " (" << avg_user_upvote_count << " up, " << avg_user_downvote_count << " down)" << std::endl;
	for (Votes_Sparse_Matrix::const_iterator m_it = g_user_sr_votes.begin(); m_it != g_user_sr_votes.end(); ++m_it) {
		ID user_id = (*m_it).first;
		const Votes_Sparse_Vector &sr_votes = (*m_it).second;
		Count user_vote_count = 0;
		for (Votes_Sparse_Vector::const_iterator v_it = sr_votes.begin(); v_it != sr_votes.end(); ++v_it) {
			Vote_Counts vote_counts = (*v_it).second;
			user_vote_count += std::get<TOTAL>(vote_counts);
		}
		if (user_vote_count == max_user_vote_count) {
			g_output_file << "A user who placed the maximum " << max_user_vote_count << " votes is: #" << user_id << " (" << g_user_id_hashes[user_id] << ")" << std::endl;
			break;
		}
	}
	g_output_file << std::endl;
}

void count_sr_vote_counts() {
	Vote_Count_Counts sr_vote_count_counts;

	g_output_file << "Counting subreddits' vote counts..." << std::endl;
	clock_t start_clock = clock();
	for (Votes_Sparse_Matrix::const_iterator m_it = g_sr_user_votes.begin(); m_it != g_sr_user_votes.end(); ++m_it) {
		const Votes_Sparse_Vector &user_votes = (*m_it).second;
		Count sr_upvote_count = 0, sr_downvote_count = 0, sr_vote_count = 0;
		for (Votes_Sparse_Vector::const_iterator v_it = user_votes.begin(); v_it != user_votes.end(); ++v_it) {
			Vote_Counts vote_counts = (*v_it).second;
			sr_upvote_count += std::get<UP>(vote_counts);
			sr_downvote_count += std::get<DOWN>(vote_counts);
			sr_vote_count += std::get<TOTAL>(vote_counts);
		}
		std::get<UP>(sr_vote_count_counts[sr_upvote_count])++;
		std::get<DOWN>(sr_vote_count_counts[sr_downvote_count])++;
		std::get<TOTAL>(sr_vote_count_counts[sr_vote_count])++;
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done counting subreddits' vote counts in " << seconds << " seconds" << std::endl;

	std::string sr_vote_count_counts_filename("sr_vote_count_counts.tsv");
	std::ofstream sr_vote_count_counts_file(g_directory + sr_vote_count_counts_filename);
	g_output_file << "Writing " << sr_vote_count_counts_filename << "..." << std::endl;
	start_clock = clock();
	sr_vote_count_counts_file.precision(g_precision);
	sr_vote_count_counts_file << "sr_vote_count\tnum_up\tnum_down\tnum_total" << std::endl;
	Count min_sr_vote_count = sr_vote_count_counts.begin()->first;
	Count max_sr_vote_count = sr_vote_count_counts.rbegin()->first;
	Statistic avg_sr_upvote_count = 0.0, avg_sr_downvote_count = 0.0, avg_sr_vote_count = 0.0;
	for (Count sr_vote_count = 0; sr_vote_count <= max_sr_vote_count; sr_vote_count++) {
		Vote_Counts ns_srs = sr_vote_count_counts[sr_vote_count];
		Count n_up_srs, n_down_srs, n_srs;
		std::tie(n_up_srs, n_down_srs, n_srs) = ns_srs;
		if (n_srs > 0) {
			sr_vote_count_counts_file << sr_vote_count << "\t" << n_up_srs << "\t" << n_down_srs << "\t" << n_srs << std::endl;
		}
		avg_sr_upvote_count += ((Statistic)sr_vote_count * (Statistic)n_up_srs) / (Statistic)g_sr_post_counts.size();
		avg_sr_downvote_count += ((Statistic)sr_vote_count * (Statistic)n_down_srs) / (Statistic)g_sr_post_counts.size();
		avg_sr_vote_count += ((Statistic)sr_vote_count * (Statistic)n_srs) / (Statistic)g_sr_post_counts.size();
	}
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << sr_vote_count_counts_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << "Each subreddit has placed in it " << min_sr_vote_count << " to " << max_sr_vote_count << " votes, average " << avg_sr_vote_count << " (" << avg_sr_upvote_count << " up, " << avg_sr_downvote_count << " down)" << std::endl;
	for (Votes_Sparse_Matrix::const_iterator m_it = g_sr_user_votes.begin(); m_it != g_sr_user_votes.end(); ++m_it) {
		ID sr_id = (*m_it).first;
		const Votes_Sparse_Vector &user_votes = (*m_it).second;
		Count sr_vote_count = 0;
		for (Votes_Sparse_Vector::const_iterator v_it = user_votes.begin(); v_it != user_votes.end(); ++v_it) {
			Vote_Counts vote_counts = (*v_it).second;
			sr_vote_count += std::get<TOTAL>(vote_counts);
		}
		if (sr_vote_count == max_sr_vote_count) {
			g_output_file << "A subreddit which has placed in it the maximum " << max_sr_vote_count << " votes is: #" << sr_id << " (" << g_sr_id_hashes[sr_id] << ")" << std::endl;
			break;
		}
	}
	g_output_file << std::endl;
}

void count_post_vote_counts() {
	Vote_Count_Counts post_vote_count_counts;

	g_output_file << "Counting posts' vote counts..." << std::endl;
	clock_t start_clock = clock();
	for (Votes_Sparse_Matrix::const_iterator m_it = g_user_sr_votes.begin(); m_it != g_user_sr_votes.end(); ++m_it) {
		const Votes_Sparse_Vector &sr_votes = (*m_it).second;
		for (Votes_Sparse_Vector::const_iterator v_it = sr_votes.begin(); v_it != sr_votes.end(); ++v_it) {
			Vote_Counts vote_counts = (*v_it).second;
			Count post_upvote_count, post_downvote_count, post_vote_count;
			std::tie(post_upvote_count, post_downvote_count, post_vote_count) = vote_counts;
			std::get<UP>(post_vote_count_counts[post_upvote_count])++;
			std::get<DOWN>(post_vote_count_counts[post_downvote_count])++;
			std::get<TOTAL>(post_vote_count_counts[post_vote_count])++;
		}
	}
	clock_t end_clock = clock();
	Statistic seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done counting posts' vote counts in " << seconds << " seconds" << std::endl;

	std::string post_vote_count_counts_filename("post_vote_count_counts.tsv");
	std::ofstream post_vote_count_counts_file(g_directory + post_vote_count_counts_filename);
	g_output_file << "Writing " << post_vote_count_counts_filename << "..." << std::endl;
	start_clock = clock();
	post_vote_count_counts_file.precision(g_precision);
	post_vote_count_counts_file << "post_vote_count\tnum_up\tnum_down\tnum_total" << std::endl;
	Count min_post_vote_count = post_vote_count_counts.begin()->first;
	Count max_post_vote_count = post_vote_count_counts.rbegin()->first;
	Statistic avg_post_upvote_count = 0.0, avg_post_downvote_count = 0.0, avg_post_vote_count = 0.0;
	for (Count post_vote_count = 0; post_vote_count <= max_post_vote_count; post_vote_count++) {
		Vote_Counts ns_posts = post_vote_count_counts[post_vote_count];
		Count n_up_posts, n_down_posts, n_posts;
		std::tie(n_up_posts, n_down_posts, n_posts) = ns_posts;
		if (n_posts > 0) {
			post_vote_count_counts_file << post_vote_count << "\t" << n_up_posts << "\t" << n_down_posts << "\t" << n_posts << std::endl;
		}
		avg_post_upvote_count += ((Statistic)post_vote_count * (Statistic)n_up_posts) / (Statistic)g_post_id_hashes.size();
		avg_post_downvote_count += ((Statistic)post_vote_count * (Statistic)n_down_posts) / (Statistic)g_post_id_hashes.size();
		avg_post_vote_count += ((Statistic)post_vote_count * (Statistic)n_posts) / (Statistic)g_post_id_hashes.size();
	}
	end_clock = clock();
	seconds = seconds_between(start_clock, end_clock);
	g_output_file << "Done writing " << post_vote_count_counts_filename << " in " << seconds << " seconds" << std::endl;
	g_output_file << "Each post has placed on it " << min_post_vote_count << " to " << max_post_vote_count << " votes, average " << avg_post_vote_count << " (" << avg_post_upvote_count << " up, " << avg_post_downvote_count << " down)" << std::endl;
	g_output_file << std::endl;
}

int main() {
	std::srand(g_random_seed);
	g_output_file.precision(g_precision);
	clock_t total_start_clock = clock();
	read_votes_file();
	write_id_hash_maps();
	calc_user_sr_affinities();
	count_user_post_counts();
	count_sr_post_counts();
	count_user_sr_counts();
	count_sr_user_counts();
	count_user_vote_counts();
	count_sr_vote_counts();
	count_post_vote_counts();
	clock_t total_end_clock = clock();
	Statistic total_seconds = seconds_between(total_start_clock, total_end_clock);
	g_output_file << "Total elapsed time: " << total_seconds << " seconds" << std::endl;
	return 0;
}
