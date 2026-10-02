#include <moonshot/state_packet.hh>
#include <cassert>
int main(){
 using namespace cellerator::experimental::moonshot;
 assert(sizeof(tile_payload<float,16>)==1024);
 assert(alignof(tile_payload<float,16>)==32);
 coordinate_ref a{1,0,1},b{2,0,1};
 assert(a.actor!=b.actor); // Same local slot index does not identify same state.
}
