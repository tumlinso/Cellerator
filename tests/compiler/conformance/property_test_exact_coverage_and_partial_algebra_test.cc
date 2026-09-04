#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
int main(){for(int n=1;n<=128;++n){std::vector<int>canonical(n),count(n),partial(n);std::iota(canonical.begin(),canonical.end(),0);for(int p=0;p<4;++p)for(int i=p;i<n;i+=4){++count[i];partial[i]+=i;}assert(std::all_of(count.begin(),count.end(),[](int x){return x==1;}));assert(partial==canonical);auto bad=count;bad[n/2]=2;assert(std::any_of(bad.begin(),bad.end(),[](int x){return x!=1;}));std::vector<int>halo=canonical;halo.insert(halo.end(),canonical.begin(),canonical.begin()+std::min(n,3));std::sort(halo.begin(),halo.end());halo.erase(std::unique(halo.begin(),halo.end()),halo.end());assert(halo==canonical);} }
