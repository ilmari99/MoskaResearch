#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>

void combine_files(const char *path, const char *output, int max_num_files) {
    FILE *f;
    DIR *dir;
    struct dirent *ent;
    char file_path[1024];
    char data[1024];
    int fileno = 0;
    // Open the output file for writing
    f = fopen(output, "w");
    if (f == NULL) {
        perror("Error opening output file");
        return;
    }

    // Open the directory specified by path
    dir = opendir(path);
    if (dir == NULL) {
        perror("Error opening directory");
        return;
    }
    // Loop through all the files in the directory
    while ((ent = readdir(dir)) != NULL && fileno < max_num_files) {
        // Construct the full path of the file
        strcpy(file_path, path);
        strcat(file_path, ent->d_name);

        // Open the file for reading
        FILE *f2 = fopen(file_path, "r");
        if (f2 == NULL) {
            perror("Error opening file");
            continue;
        }
        if (fileno%100 == 0){
            printf("%d files combined.\n",fileno);
        }
        fileno++;
        // Read the contents of the file and write it to the output file
        while (fgets(data, sizeof(data), f2) != NULL) {
            fputs(data, f);
        }
        // Add a newline to the output file

        fputs("\n", f);

        // Close the file
        fclose(f2);
    }

    // Close the directory
    closedir(dir);

    // Close the output file
    fclose(f);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <input folder> <output file> <max_num_files>\n", argv[0]);
        return 1;
    }

    // Call the combine_files function with the input folder and output file name
    combine_files(argv[1], argv[2], atoi(argv[3]));

    return 0;
}
