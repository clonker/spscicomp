#include "HiddenMarkovModel.h"
#include <iostream>
#include <fstream>

#define EXIT_SUCCESS     0
#define EXIT_USAGE       1
#define EXIT_FILE_SYNTAX 2
#define EXIT_UNKNOWN     3

int main(int argc, char *argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    return EXIT_USAGE;
  }
  spscicomp::hidden_markov_model<float> hmm;
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    file.open(argv[1]);
    file >> hmm;
    file.close();

    std::cout << "Successfully read hmm file:\n";
    std::cout << hmm << std::endl;

  } catch (spscicomp::bad_file_syntax e) {
    std::cerr << "Error: bad syntax in file '" << argv[1] << "': "
              << e.what() << std::endl;
    return EXIT_FILE_SYNTAX;
  } catch (std::ifstream::failure e) {
    std::cerr << "Could not open file '" << argv[1] << "': " << e.what() << std::endl;
    return EXIT_UNKNOWN;
  }
  return EXIT_SUCCESS;
}