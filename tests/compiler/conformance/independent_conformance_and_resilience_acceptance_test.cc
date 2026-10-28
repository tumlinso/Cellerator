#include <cassert>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
int main(){std::ifstream f("tests/compiler/conformance/part_one_coverage_matrix.json");assert(f);std::string s((std::istreambuf_iterator<char>(f)),{});assert(s.find("CELLERATOR-CCP1-CONFORMANCE/1")!=std::string::npos);for(const auto*c:{"source","ceir","profile","planner","realization","backend","lto","extension","sdk","celleratord"}){auto p=s.find(std::string("\"")+c+"\"");assert(p!=std::string::npos&&s.find("\"passed\"",p)!=std::string::npos);}assert(s.find("\"unexplained_regressions\": []")!=std::string::npos);}
