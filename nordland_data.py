import os, time
import struct
import numpy as np
from PIL import Image

cwd = (os.path.dirname(os.path.realpath(__file__)))
season_labels = ["summer", "winter", "fall", "spring"]

xdim = 48
ydim = 48

num_imgs_train = 24600
num_imgs_test = 2300

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py.
"""
def show(image):
    """
    Render a given image.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

    
# Given an index and a season, this method loads an image
def load_image(image_index, path_im, dataset="training", block_size=1150):

    images = []
    labels = []
    for i in range(image_index.shape[0]):
        
        idx = image_index[i]
        section_index = np.digitize(idx, [0, block_size])
        path = f'{path_im}/section{section_index}/{idx}.png'
        
        image = np.array(Image.open(path).resize((xdim,ydim)).convert('L'))/255.
        label = idx
        
        images.append(image)
        labels.append(label)

    image_array = np.stack(images)

    return (image_array, labels)


def generate_inputs(season, train_size=5000, test_size=500, plot=False, seed=None, dataset='train', data_prefix='data/Partitioned_Nordland_Dataset_lowres/Dataset_images/'):

    if seed == None:
        np.random.seed(int((time.time() * 1000000000 ) % (2**32 - 1)))
    else:
        np.random.seed(seed)

    if dataset == 'train':
        img_idx = np.random.randint(0, high=num_imgs_train, size=train_size)
    elif dataset == 'test':
        img_idx = np.random.randint(0, high=num_imgs_test, size=test_size)

    num_imgs = None
    if dataset == 'train':
        imgs, lbls = load_image(image_index=img_idx, path_im=f'{data_prefix}/train/{season}_images_train', block_size=12837)
        num_imgs = imgs.shape[0]
    elif dataset == 'test':    
        imgs, lbls = load_image(image_index=img_idx, path_im=f'{data_prefix}/test/{season}_images_test')
        num_imgs = imgs.shape[0]

    if plot:
        show(imgs[0])

    return imgs, lbls




def test():
	parser = argparse.ArgumentParser(description='Test Single-View Place Recognition.')
	parser.add_argument('--dataset',required=True,type=str,help="Path to Partitioned Nordland. e.g. for given/path/ it should contain given/path/test ")
	parser.add_argument('--input_season',default=0,type=int,required = False, help="[default = 0] 0 = summer, 1 = winter, 2 = fall, 3 = spring.")
	parser.add_argument('--reference_season',default=1,type=int,required = False, help="[default = 1] 0 = summer, 1 = winter, 2 = fall, 3 = spring.")
	args = parser.parse_args()
	# Seasons to compare. Change as desired.
	# First season is the input one. Second season is the reference season.
	input_season = args.input_season     # 0 = summer, 1 = winter, 2 = fall, 3 = spring.
	reference_season = args.reference_season # 0 = summer, 1 = winter, 2 = fall, 3 = spring.


	path_im = [ os.path.join(args.dataset,sp) for sp in ['test/summer_images_test/', 
		                                             'test/winter_images_test/',
		                                             'test/fall_images_test/',
		                                             'test/spring_images_test/']]
	
		
	# Spiking neuron autoencoder reconstruction
	net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)

	# Variables
	count = 0
	num_neighbours = 5 # Number of neighbours
	precision_at_k = np.zeros(num_neighbours) # Fraction of correct matches
	at_least_one_at_k = np.zeros(num_neighbours) # Fraction of matches with at least one correct place
	closest_places = np.zeros((test_size, num_neighbours)) # Matrix to save the closest features labels
	all_matches = np.zeros((test_size, test_size)) # Matrix to save all the features labels ordered by distance

	# Feature extraction
	print(" Extracting features...this may take a while...")
	input_features, input_labels = extract_features(test_size, input_season, net, path_im)
        # Dims: [number of images, feature size], [number of images]
        
	reference_features, reference_labels = extract_features(test_size, reference_season, net, path_im)
	print(" Features extracted...")
	print(" Initializing comparison...")

	for i in range(test_size):
	    closest_places_labels = closestFeatures(input_features[i], reference_features)
	    closest_places[i,:] = closest_places_labels[0:num_neighbours]
	    all_matches [i,:] = closest_places_labels
	    #print(" Closest reference places: ", reference_labels[closest_places_labels[0:num_neighbours]])
	    for j in range(num_neighbours):
                number_of_votes = numberOfCorrectMatches(j+1, reference_labels[closest_places_labels[0:num_neighbours]], input_labels[i])
                precision_at_k [j] += number_of_votes
                if ( number_of_votes >= 1 ):
                    at_least_one_at_k [j] += 1

	# Output the metrics				
	print("Evaluated Neural network: ", caffe_model_file_path)
	print("Input season: ", string_seasons[input_season])
	print("Reference season: ", string_seasons[reference_season])

	# Fraction of correct matches 
	precision_at_k = precision_at_k/test_size
	at_least_one_at_k = at_least_one_at_k/test_size
	# Percentages are made by considering all the input places
	for neighbour in range(len(precision_at_k )):
		precision_at_k [neighbour] = precision_at_k[neighbour]/(neighbour+1)
		
	print("Fraction of correct matches: ", precision_at_k[0]*100.0, "%")
	"""
	print(" Fraction of correct matches (considering 1 to 5 closest neighbours) is: ")
	print(precision_at_k)
	print(" Fraction of matches (considering 1 to 5 closest neighbours) with at least one correct match in them: ")
	print(at_least_one_at_k)
	"""
	print("")



