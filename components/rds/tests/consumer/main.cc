import cellerator.rds;

int main() {
    cellerator::rds::RdsFile file;
    return file.info().columns != 0;
}
