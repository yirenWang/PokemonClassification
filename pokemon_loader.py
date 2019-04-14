import torch
import numpy as np
import cv2
import os
import io
import gc
import glob
from tqdm import tqdm

# Loader for one-shot-pokemon dataset for one-shot learning
class PokemonLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True, matches=False, hard=False, transform=None, n=0, *arg, **kw):
        # Transforms to apply to the images
        self.transform = transform
        
        # Root directory of the images
        self.root_dir = root_dir
        
        # Tells if in training mode ()
        self.train = train
        
        # If true, matches should be given instead of triplets
        # Useful for validation and test
        self.matches = matches
        
        # Tells if the harder dataset (pokemon-tcg) should be used
        self.hard = hard
        
        # Number of triplets / matches to generate
        self.n = n
    
        # We train on 1st gen Pokémon images, validate on 2nd gen, and test on 3rd gen
        if self.train:
            self.labels = list(range(1, 152))
            print(f"Generating {self.n} triplets")
            self.triplets = self.generate_triplets(self.labels, self.n)
        elif self.matches:
            self.labels = list(range(252, 387))
            print(f"Generating {self.n} matches")
            self.matches = self.generate_matches(self.labels, self.n)
        else:
            self.labels = list(range(152, 252))
            print(f"Generating {self.n} matches")
            self.matches = self.generate_matches(self.labels, self.n)

    def generate_matches(self, labels, num_matches):
        """
        Generate image pairs for validation and test.
        Pairs have a 50% chance to be two images of the same Pokémon,
        and a 50% chance to be different.
        """
        matches = []
        n_classes = len(self.labels)

        for x in tqdm(range(num_matches)):
            # Choose first Pokémon
            c1 = np.random.randint(0, n_classes)

            # About half the pairs are of the same Pokémon
            same = x < num_matches / 2
            if not same:
                c2 = np.random.randint(0, n_classes)
                while c1 == c2:
                    c2 = np.random.randint(0, n_classes)   

                first = np.random.randint(0, 2)
                path1 = (self.labels[c1], first)

                second = np.random.randint(0, 2)
                path2 = (self.labels[c2], second)
           
            else:
                first = np.random.randint(0, 2)
                path1 = (self.labels[c1], first)
                path2 = (self.labels[c1], 1-first)
            # HACK: as path1 and path2 are tuples, the last element has to be a 
            # tuple (even if we just need a boolean).
            matches.append([path1, path2, (same, same)])

        return torch.LongTensor(np.array(matches))

    def generate_triplets(self, labels, num_triplets):
        """
        Generate image triplets (A, P, N) to train the network.
        The two pokémon used for each triplet are randomly chosen.
        """
        triplets = []
        n_classes = len(self.labels)

        for x in tqdm(range(num_triplets)):
            # Choose first Pokémon
            c1 = np.random.randint(0, n_classes)
            
            # Choose second Pokémon
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)

            # Choose which image of first Pokémon is anchor
            first = np.random.randint(0, 2)
            path11, path12 = (self.labels[c1], first), (self.labels[c1], 1-first)

            # Choose an image of second Pokémon
            first = np.random.randint(0, 2)
            path21 = (self.labels[c2], first)

            triplets.append([path11, path12, path21])

        return torch.LongTensor(np.array(triplets))
            
    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img)
            return img
        
        # Get path to the image from (label, image folder id) code
        def path_of(code):
            number = code[0]
            if code[1] == 0:
                return f"pokemon-a/{number}.png"
            if code[1] == 1:
                return f"pokemon-b/{number}.jpg"
                
        if not self.train:
            
            # We're evaluating, we have a match 
            # (image 1 code, image 2 code, boolean which says if same)
            m = self.matches[index]
            
            img_name_1 = os.path.join(self.root_dir, path_of(m[0]))
                          
            possible_image_names = []
            if self.hard:
                possible_image_names = glob.glob(os.path.join(self.root_dir, "pokemon-tcg-images/", f"{m[1][0]}-*"))
            if (len(possible_image_names) > 0):
                i = np.random.randint(0, len(possible_image_names))
                img_name_2 = possible_image_names[i]
                
                while len(possible_image_names) > 1 and img_name_1 == img_name_2:
                    #  Make sure different images are chosen
                    i = np.random.randint(0, len(possible_image_names))
                    img_name_2 = possible_image_names[i]
            else:
                # If not in hard mode OR there are no pokemon-tcg images for the Pokémon
                # Then choose one of pokemon-a or pokemon-b images
                img_name_2 = os.path.join(self.root_dir, path_of(m[1]))
                                             
            # Load the images and pre-process them                          
            img_1 = cv2.cvtColor(cv2.imread(img_name_1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img_2 = cv2.cvtColor(cv2.imread(img_name_2, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
            img_1 = transform_img(img_1)
            img_2 = transform_img(img_2)

            return img_1, img_2, m[2]

        # We're training: we have triplets
        t = self.triplets[index]
        
        # Load the images and pre-process them
        img_name_a = os.path.join(self.root_dir, path_of(t[0]))
        img_a = cv2.cvtColor(cv2.imread(img_name_a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_a = transform_img(img_a)
        
        img_name_p = os.path.join(self.root_dir, path_of(t[1]))
        img_p = cv2.cvtColor(cv2.imread(img_name_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_p = transform_img(img_p)
        
        img_name_n = os.path.join(self.root_dir, path_of(t[2]))
        img_n = cv2.cvtColor(cv2.imread(img_name_n, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_n = transform_img(img_n)
        
        return img_a, img_p, img_n
    
    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)
