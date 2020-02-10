## Description
This script generates high resolution text for The Legend of Zelda Ocarina of
Time 3D, to be used with the Citra 3DS emulator. Citra includes support for
replacing low resolution textures with high resolution alternatives. In most
games, this functionality makes it straightforward to obtain high resolution
text by replacing a few font textures. However, Ocarina of Time has an unusual
text implementation in which each time a text string is displayed an image
including one instance of each unique character in the string is created. This
behavior is presumably inherited from the Nintendo 64 version, on which it may
have been useful for conserving memory. The script works around this behavior
by:
1. Identifying text image files in the dump directory to which Citra saves
   textures
2. Separating each text image into its individual characters and identifying
   them
3. Writing a new high resolution text image in the load directory from which
   Citra loads replacement textures
The script accumulates text images as the game is played. The first time a text
string is displayed it will be low resolution, but the next time that same
string is loaded and displayed it will be replaced with a high resolution
version. Playing through the complete game should therefore yield a complete
high resolution text package.

**Unfortunately**, the North American and European localizations have a
behaviour that sharply limits the usefulness of this script: they do not fully
clear the text image between uses. If string A with 15 unique characters is
displayed, followed by string B with 20 unique characters, and then string A
again, two unique text images will be generated for string A. The first will
include string A's 15 unique characters, and the second will include string A's
15 unique characters followed by string B's 16th through 20th characters. Thus,
the text image of a string depends on the contents of the strings viewed
previously, grossly increasing the number of potential text images. This is
difficult to work around without modifying either the internals of the game or
Citra, and limits the usefulness of the script for European languages. It is
still fully functional for these languages, but the chance of encountering high
resolution text is relatively slim. I do not know if this limitation applies to
the Japanese or Korean localizations.

**Fortunately**, the Chinese localization, for which this script was originally
intended, clears the text image between usages, making it viable to generate a
complete high resolution text pack for this language. The script is
particularly useful for Chinese, as the original characters' 16x16 pixel
resolution reduces many complex characters to blurs. This script has been used
to generate a simplified Chinese high resolution text package. It may still be
run while playing the Chinese localization to capture any additional text
missing from the released package

## Usage

## License
