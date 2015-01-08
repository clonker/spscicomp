import numpy

models = dict();

models['hmm_1'] =  (
	numpy.array(
		[[0.8, 0.2, 0.0],
		 [0.0, 0.2, 0.8],
		 [1.0, 0.0, 0.0]],
	),
	numpy.array(
		[[1.0, 0.0],
		 [0.0, 1.0],
		 [0.0, 1.0]],
	),
	numpy.array(
		[0.5, 0.5, 0.0],
	)
)
models['equi32'] = (
    numpy.array(
        [[ 0.333, 0.333, 0.333 ],
         [ 0.333, 0.333, 0.333 ],
         [ 0.333, 0.333, 0.333 ]]),

    numpy.array(
        [[ 0.5, 0.5 ],
         [ 0.5, 0.5 ],
         [ 0.5, 0.5 ]]),

    numpy.array([ 0.333, 0.333, 0.333 ])
)


def get_models():
	return models