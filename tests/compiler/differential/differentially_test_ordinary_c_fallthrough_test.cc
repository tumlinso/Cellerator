#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>
template<class T> T fold(std::vector<T> x){std::sort(x.begin(),x.end());return std::accumulate(x.begin(),x.end(),T{});}
struct value{int x; explicit operator bool()const{return x!=0;}};
int main(){std::vector<int>v{5,1,3,2,4};auto lambda=[](auto x){return x*x;};std::tuple<int,std::string>t{fold(v),"fallthrough"};value condition{1};if(condition)std::cout<<std::get<1>(t)<<':'<<std::get<0>(t)<<':'<<lambda(4)<<'\n';return 0;}
