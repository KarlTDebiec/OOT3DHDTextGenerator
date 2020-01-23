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
        #       chars/assignment (N str)
        #       chars/confirmation (N bool)
        #       chars/image (Nx256 uint8)
        #   Reconstruct data structures
        #       char_assignments (dict)
        #           Keys are char images (ndarray -> bytes)
        #           Values are tuple of (assignment (str), confirmation (bool))

        # Load image cache
        #   HDF5 file with three tables:
        #       images/assignment (N str)
        #       images/confirmation (N bool)
        #       images/filename (Nx36 str)
        #       images/image (Nx65536 uint8 ndarray)
        #   Reconstruct data structures
        #       confirmed_images (dict):
        #           Keys are filenames
        #

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
