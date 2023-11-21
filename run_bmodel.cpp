#include "sail/engine.h"
#include "sail/tensor.h"
using namespace std;
using namespace sail;

static char* fp = "espcn_ex.fp32.bmodel";

int main(int argc, char* argv[]) {
  int tpu_id = 0;
  Engine engine(tpu_id);
  printf("engine: %p", engine.get_input_tensor());
  Handle handle = engine.get_handle();
  printf("handle: %p", handle);
  engine.load(fp);
  printf("bmodel: %s", fp);
  vector<string> graph_names = engine.get_graph_names();
  engine.set_io_mode(graph_names[0], DEVIO);
  
  Tensor x = Tensor();
  x.memory_set(0xF0);
  printf("tensor: %p", x);
}
