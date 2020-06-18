#include <stdio.h>

// Return true if chunk x,z is a slime chunk, false otherwise. Use s as seed.
// https://minecraft.gamepedia.com/Slime
// https://docs.oracle.com/javase/7/docs/api/java/util/Random.html
// http://developer.classpath.org/doc/java/util/Random-source.html
__device__ bool isSlimeChunk(long long s, long long x, long long z) {
  unsigned long long seed = (s +
      (int) (x * x * 0x4c1906LL) +
      (int) (x * 0x5ac0dbLL) +
      (int) (z * z) * 0x4307a7LL +
      (int) (z * 0x5f24fLL)
      ^ 0x5E434E432LL) & ((1LL << 48) - 1);

  int bits, val;
  do {
    seed = (seed * 0x5DEECE66DLL + 0xBLL) & ((1LL << 48) - 1);
    bits = (int)(seed >> 17);
    val = bits % 10;
  } while (bits - val + 9 < 0);

  return val == 0;
}

// Return the square root of x
__device__ __host__ int sqrt(int x) {
  if (x == 0 || x == 1) return x;
  int i = 1, result = 1;
  while (result <= x) {
    ++i;
    result = i*i;
  }
  return i-1;
}

// Calculate the slime chunks in the initial row. This is different because the old values cannot be reused.
// Use x as the x coordinate, s as the seed and store the result in row.
// Each bit in row is one chunk, each int in row is one z coordinate.
__global__ void setInitialX(long long s, int x, int* row) {
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  int current = 0;
  
  for (int i = 0; i < 32; ++i) {
    current |= isSlimeChunk(s, x+i, z - 1875000) << (31-i);
  }

  row[z] = current;
}

// Same as setInitialX, except it reuses a few chunks (which were already calculated in the previous setX/setInitialX call).
__global__ void setX(long long s, int x, int* row) {
  int z = blockIdx.y * blockDim.y + threadIdx.y;
  int current = row[z] << 16; // 17 chunks are tested at a time so 16 need to be included in the next search again
                              // 32-16 == 16

  for (int i = 16; i < 32; ++i) {
    current |= isSlimeChunk(s, x+i, z - 1875000) << (31-i);
  }

  row[z] = current;
}

// Count the amount of "1" bits in x and return it.
// This is only better if most bits are 0, which is the case for slime chunks (9/10 is 0 and 1/10 is 1).
// https://en.wikipedia.org/wiki/Hamming_weight
// TODO see if a lookup table is faster and if it is, implement it
__device__ int popcount(int x) {
  int count;
  for (count = 0; x; count++) {
    x &= x - 1;
  }
  return count;
}

// Count how many blocks are loaded and in a slime chunk at the same time.
// Take the slime chunks from row and return the result in n.
// currentMax is the current highest value. If the value cannot exceed this, the calculation will be stopped earlier to save time.
__global__ void count(int* row, unsigned short* n, int currentMax, short* sb, short* sbs) {
  int x = blockIdx.x * blockDim.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;

  // Avoid going too high and trying to access not available memory
  if (z >= 3750000) return;

  // Load the chunks from row
  int chunks[17];
  for (int i=0; i<17; ++i) {
    chunks[i] = row[z+i] >> 15-x & 0x1FFFF;
  }

  // First count all in a 17x17 area to simplify, if this doesn't succeed, don't check all different locations within that chunk.
  short sum = 0;
  for (int i=0; i<17; ++i) {
    sum += popcount(chunks[i]);
  }
  if (sum*256 <= currentMax) {
    n[16*z+x] = 0;
    return;
  }

  // Second simplification, exclude the blocks that are not in any of the spawnblocks and test again
  int cl[289]; // chunklist, 1 if a chunk is a slime chunk, otherwise 0 for each chunk in a 17x17 area
  for (int i=0; i<17; ++i) {
    for (int j=0; j<17; ++j) {
      cl[i*17+j] = chunks[i] >> (16-j) & 1;
    }
  }
  sum = 0;
  for (int i=0; i<17; ++i) {
    for (int j=0; j<17; ++j) {
      sum += cl[i*17+j]==1 ? sbs[i*17+j] : 0;
    }
  }
  if (sum <= currentMax) {
    n[16*z+x] = 0;
    return;
  }

  // Calculate different rates for all heights and positions inside the chunk
  short heighest = 0;
  for (int X=0; X<16; ++X) {
    for (int Z=0; Z<16; ++Z) {
      for (int y=0; y<24; ++y) {
        sum = 0;
        for (int i=0; i<17; ++i) {
          for (int j=0; j<17; ++j) {
            sum += cl[i*17+j]==1 ? sb[x*110976+Z*6936+y*289+i*17+j] : 0;
          }
        }
        if (sum > heighest) {
          heighest = sum;
        }
      }
    }
  }

  if (heighest > currentMax) {
    //printf("%u,%u - %u\n",x,z,heighest);
    n[16*z+x] = heighest;
  } else {
    n[16*z+x] = 0;
  }

  return;
}

// Fill f with the amount of blocks within spawn radius (24 < x <= 128) in each chunk, for each x and z coordinate in the chunk and 24 y coordinates
// it uses 17x17 chunks, not 16x16
__global__ void calcSpawnableBlocks(short int* f) {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int x = thread/384;
  int y = thread%384/16;
  int z = thread%16;
  int i = x*110976+z*6936 + y*289;
//  for (int xx=-128; xx<129; ++xx) {
//    for (int zz=-128; zz<129; ++zz) {
  for (int xx=-128; xx<145; ++xx) {
    for (int zz=-128; zz<145; ++zz) {
      int t = sqrt((x-xx)*(x-xx) + (z-zz)*(z-zz) + y*y);
      if (24 < t && t <= 128) {
//        f[i + (xx+128)/17*17+(zz+128)/17]++;
        f[i + (xx+128)/16*17+(zz+128)/16]++;
      }
    }
  }
}

// Count blocks if slime can spawn there at any x, y, or z position of the player
void calcSecondSpawnableBlocks(short* a) {
  for (int i=0; i<17*17; ++i) a[i] = 0;
  for (int xx=-128; xx<145; ++xx) {
    for (int zz=-128; zz<145; ++zz) {
      bool temp = false;
      for (int x=0; x<16 && !temp; ++x) {
        for (int z=0; z<16 && !temp; ++z) {
          if (sqrt((x-xx)*(x-xx) + (z-zz)*(z-zz)) <= 128) temp = true;
        }
      }
      if (temp) a[(xx+128)/16*17 + (zz+128)/16]++;
    }
  }
}

// Find the highest amount of slime chunks in an area and the location of those slime chunks
// c (count) is the amount of slime chunks in an area and l (location) is the location of those slime chunks.
// the highest 1024 will be returned in the first 1024 items in count and location
// http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
__global__ void findHighest(unsigned short *c, unsigned short *o, int *l) {
  __shared__ short scount[1024];
  __shared__ int slocation[1024];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= 59999232) return;
  scount[tid] = c[i];
  slocation[tid] = i;
  __syncthreads();

  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      if (scount[tid+s] > scount[tid]) {
        scount[tid] = scount[tid+s];
        slocation[tid] = slocation[tid+s];
      }
    }
    __syncthreads();
  }

  if(tid==0) {
    o[blockIdx.x] = scount[0];
    l[blockIdx.x] = slocation[0];
  }
}

// Inefficient function to move the location and count of the highest chunk to index 0 of c and l
void findHighestCpu(unsigned short* c, int* l, unsigned short* oc, int* cl) {
  cudaMemcpy(c, oc, 58608 * sizeof(short), cudaMemcpyDeviceToHost);
  cudaMemcpy(l, cl, 58608 * sizeof(int), cudaMemcpyDeviceToHost);
  
  unsigned short heighest = 0;
  for (int i=0; i<58608; ++i) heighest = c[i]>heighest ? c[i] : heighest;
  for (int i=0; i<58608; ++i) {
    if (c[i] == heighest) {
      c[0] = c[i];
      l[0] = l[i];
      return;
    }
  }
}

void printError() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
}

int main() {
  int *chunks; // Each bit represents a chunk. 1 if the chunk is a slime chunk, otherwise 0
  cudaMalloc((void**)&chunks, sizeof(int) * 3750912);
  unsigned short *chunkCount; // The amount of blocks slimes can spawn on in a 17x17 area centered around the player
  cudaMalloc((void**)&chunkCount, sizeof(short) * 3750000 * 16);
  short int *spawnBlocks; // A list with the amount of spawnable blocks in each chunk for each position in the chunk and each height
  cudaMalloc((void**)&spawnBlocks, sizeof(short int) * 17*17 * 16*16*24);
  cudaMemset(spawnBlocks, 0, sizeof(short int) * 17*17 * 16*16*24);
  int *chunkLocation; // The location of the chunk corresponding to the count in chunkCount. Format is z*16+x
  cudaMalloc((void**)&chunkLocation, sizeof(int) * 3663 * 16);
  unsigned short *outCount; // The output of the findHeighest function
  cudaMalloc((void**)&outCount, sizeof(short) * 3663 * 16);
  short *spawnBlocksSecond; // Combination of all possible heights and positions of spawnBlocks
  cudaMalloc((void**)&spawnBlocksSecond, sizeof(short) * 17 * 17);

  // Host variables for outCount and chunkLocation
  unsigned short *countt = (unsigned short*) malloc(58608 * sizeof(short));
  int *location = (int*) malloc(58608 * sizeof(int));

  // Get seed, minx and maxx from user input
  long long seed = 0;
  char temp[8] = {0};
  int minx = -1875000;
  int maxx = 1875000;
  int heighest = 0;
  printf("Enter seed: ");
  scanf("%d", &seed);
  printf("Enter x start (min is -1875000) ('d' for default -1875000): ");
  scanf("%s", temp);
  if (temp[0] != 'd')
    sscanf(temp, "%d", &minx);
  printf("Enter x end (max is 1875000) ('d' for default 1875000): ");
  scanf("%s", temp);
  if (temp[0] != 'd')
    sscanf(temp, "%d", &maxx);
  if (minx < -1875000 || maxx > 1875000 || maxx < minx) {
    printf("Illegal minx or maxx, range is -1875000 to 1875000 and minx must be smaller than maxx");
    return 1;
  }
  printf("Enter heighest to begin with: ");
  scanf("%d", &heighest);

  printf("Calculating spawnable blocks\n");  
  calcSpawnableBlocks<<<6, 1024>>>(spawnBlocks);
  cudaDeviceSynchronize();

  short *sbsCpu = (short*) malloc(17*17*sizeof(short));
  printf("Calculating second spawnable blocks\n");
  calcSecondSpawnableBlocks(sbsCpu);
  cudaDeviceSynchronize();
  cudaMemcpy(spawnBlocksSecond, sbsCpu, 17*17*sizeof(short), cudaMemcpyHostToDevice);

  printf("Setting initial X\n");
  setInitialX<<<dim3(1, 3663), dim3(1, 1024)>>>(seed, minx, chunks);
  cudaDeviceSynchronize();

  printf("Counting slime chunks/blocks\n");
  count<<<dim3(16, 3663), dim3(1, 1024)>>>(chunks,chunkCount,heighest,spawnBlocks,spawnBlocksSecond);
  cudaDeviceSynchronize();

  printf("Finding heighest chunk\n");
  findHighest<<<3663 * 16, 1024>>>(chunkCount, outCount, chunkLocation);
  cudaDeviceSynchronize();

  findHighestCpu(countt, location, outCount, chunkLocation);
  //printf("Start value - start chunk: %u - %u\n", countt[0], location[0]);
  if (countt[0] > heighest) {
    printf("New heighest in first row: %u - %u\n", countt[0], location[0]);
    heighest = countt[0];
  }

  printError();

  printf("Starting computation of other chunks\n");

  //for (int i = -1874984; i < 1875000; i += 16) {
  //for (int i = -1874984; i < -1870000; i += 16) {
  for (int i = minx+16; i < maxx; i += 16) {
    setX<<<dim3(1, 3663), dim3(1, 1024)>>>(seed, i, chunks);
    cudaDeviceSynchronize();

    count<<<dim3(16, 3663), dim3(1, 1024)>>>(chunks,chunkCount,heighest,spawnBlocks,spawnBlocksSecond);
    cudaDeviceSynchronize();

    findHighest<<<3663 * 16, 1024>>>(chunkCount, outCount, chunkLocation);
    cudaDeviceSynchronize();

    findHighestCpu(countt, location, outCount, chunkLocation);
    if (countt[0] > heighest) {
      printf("New heighest value: %u - %u with i %d, absolute location: %d, %d\n", countt[0], location[0], i, location[0]%16*16+i*16, location[0]/16*16-30000000);
      heighest = countt[0];
    }

    printError();
  }

  cudaFree(chunks);
  cudaFree(chunkCount);
  cudaFree(spawnBlocks);
  cudaFree(chunkLocation);
  cudaFree(outCount);
  cudaFree(spawnBlocksSecond);
  free(countt);
  free(location);
  free(sbsCpu);
}
