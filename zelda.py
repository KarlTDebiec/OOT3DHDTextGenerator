#!/usr/bin/python

def main():
    import yaml
    from os import listdir

    # Read configuration
    with open("conf.yaml", "r") as infile:
        conf = yaml.load(infile, Loader=yaml.SafeLoader)

    for game, settings in conf.items():

        # Get settings and validate
        dump_dir = settings["Dump directory"]

        # Load character cache
        #   HDF5 file with three tables:
        #       characters/character (N str)
        #       characters/confirmation (N bool)
        #       characters/image (Nx16x16 uint8)
        #   Reconstruct data structures
        #       characters (dict)
        #           Keys are char images (bytes)
        #           Values are tuple of (assignment (str), confirmation (bool))

        # Load confirmed image cache
        #   HDF5 file with two tables:
        #       images/confirmed/filename (Nx36 str)
        #       images/confirmed/text (N str)
        #   Reconstruct data structures
        #       confirmed_images (dict):
        #           Keys are filenames (str)
        #           Values are text (str)

        # Load unconfirmed image cache
        #   HDF5 file with three tables:
        #       images/unconfirmed/filename (Nx36 str)
        #       images/unconfirmed/image (Nx256x256 uint8)
        #   Reconstruct data structures
        #       unconfirmed_images (dict):
        #           Keys are filenames (str)
        #           Values are list of character images in bytes

        # Review all existing images
        for i, image in enumerate(listdir(dump_dir)):
            print(i, image)
            # Skip if size not 256x256
            # Load image
            # Make grayscale
            # Check confirmed

        # Start watcher


if __name__ == "__main__":
    # Command-Line Arguments:
    #   Conf file
    #   Game
    #   Mode
    #
    main()
