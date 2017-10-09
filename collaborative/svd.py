import sys
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import time

g_num_users = None
g_num_srs = None
g_max_num_singular_values = None
g_test_sample_id = None

g_directory = 'C:/Users/Remy/Desktop/590/'
g_output_filename = None
g_output_file = None
g_average_affinity = 0.855973630958

g_user_sr_norm_affinities = None
g_num_affinities = 0

def read_affinities_file():
	global g_user_sr_norm_affinities, g_num_affinities
	affinities_filename = 'affinities.tsv'
	affinities_file = open(g_directory + affinities_filename, 'r')
	g_output_file.write('Reading %s; building matrix...\n' % affinities_filename)
	g_output_file.flush()
	start_clock = time.clock()
	header_line = affinities_file.readline()
	g_user_sr_norm_affinities = scipy.sparse.dok_matrix((g_num_users, g_num_srs))
	affinity_line = affinities_file.readline()
	while affinity_line:
		user_id, sr_id, affinity, num_votes, sample_id = affinity_line.split('\t')
		user_id = int(user_id)
		sr_id = int(sr_id)
		affinity = float(affinity)
		num_votes = int(num_votes)
		sample_id = int(sample_id)
		if sample_id != g_test_sample_id:
			normalized_affinity = affinity - g_average_affinity
			g_user_sr_norm_affinities[user_id, sr_id] = normalized_affinity
			g_num_affinities += 1
		affinity_line = affinities_file.readline()
	g_user_sr_norm_affinities = g_user_sr_norm_affinities.tocsc()
	end_clock = time.clock()
	seconds = end_clock - start_clock
	g_output_file.write('Done reading %s in %.12f seconds\n' % (affinities_filename, seconds))
	g_output_file.write('%d user-subreddit affinities, between %d users and %d subreddits\n' % (g_num_affinities, g_num_users, g_num_srs))
	g_output_file.write('\n')
	g_output_file.flush()

def write_svd_matrix_files():
	g_output_file.write('Calculating singular value decomposition...\n')
	g_output_file.flush()
	start_clock = time.clock()
	U, svs, Vt = scipy.sparse.linalg.svds(g_user_sr_norm_affinities, k=g_max_num_singular_values)
	Sigma = scipy.linalg.diagsvd(svs, g_max_num_singular_values, g_max_num_singular_values)
	sqrtSigma = scipy.linalg.sqrtm(Sigma)
	UsqrtSigma = U.dot(sqrtSigma)
	sqrtSigmaVt = sqrtSigma.dot(Vt)
	end_clock = time.clock()
	seconds = end_clock - start_clock
	g_output_file.write('Done calculating SVD in %.12f seconds\n' % seconds)
	g_output_file.flush()
	
	g_output_file.write('Writing SVD matrix files...\n')
	g_output_file.flush()
	start_clock = time.clock()
	
	def write_svd_matrix(svd_matrix_filename, svd_matrix):
		svd_matrix_file = open(g_directory + svd_matrix_filename, 'w')
		g_output_file.write('Writing %s...\n' % svd_matrix_filename)
		g_output_file.flush()
		svd_matrix_file.write('%d %d\n' % svd_matrix.shape)
		for i in range(svd_matrix.shape[0]):
			for j in range(svd_matrix.shape[1]):
				svd_matrix_file.write('%s' % repr(svd_matrix[i, j]))
				if j < svd_matrix.shape[1] - 1:
					svd_matrix_file.write(' ')
			svd_matrix_file.write('\n')
		svd_matrix_file.close()
		g_output_file.write('Done writing %s\n' % svd_matrix_filename)
		g_output_file.flush()
	
	svd_matrix_u_filename = 'svd_matrix_u_st.dat'
	write_svd_matrix(svd_matrix_u_filename, U)
	
	svd_matrix_sigma_filename = 'svd_matrix_sigma_st.dat'
	write_svd_matrix(svd_matrix_sigma_filename, Sigma)
	
	svd_matrix_vt_filename = 'svd_matrix_vt_st.dat'
	write_svd_matrix(svd_matrix_vt_filename, Vt)
	
	svd_matrix_usqrtsigma_filename = 'svd_matrix_usqrtsigma_st.dat'
	write_svd_matrix(svd_matrix_usqrtsigma_filename, UsqrtSigma)
	
	svd_matrix_sqrtsigmavt_filename = 'svd_matrix_sqrtsigmavt_st.dat'
	write_svd_matrix(svd_matrix_sqrtsigmavt_filename, sqrtSigmaVt)
	
	end_clock = time.clock()
	seconds = end_clock - start_clock
	g_output_file.write('Done writing SVD matrix files in %.12f seconds\n' % seconds)
	g_output_file.write('\n')
	g_output_file.flush()

def main():
	total_start_clock = time.clock()
	read_affinities_file()
	write_svd_matrix_files()
	total_end_clock = time.clock()
	total_seconds = total_end_clock - total_start_clock
	g_output_file.write('Total elapsed time: %.12f seconds\n' % total_seconds)

if __name__ == '__main__':
	g_num_users = int(sys.argv[1])
	g_num_srs = int(sys.argv[2])
	g_max_num_singular_values = int(sys.argv[3])
	g_test_sample_id = int(sys.argv[4])
	g_output_filename = 'svd_output_%02d.log' % g_test_sample_id
	g_output_file = open(g_directory + g_output_filename, 'w')
	main()
	g_output_file.close()
