int find_max(float* vect_in, int len) {
  int max = 0;
  for (int i = 0; i < len; i++) {
    if (vect_in[max] < vect_in[i]) {
      max = i;
    }
  }
  return max;
}