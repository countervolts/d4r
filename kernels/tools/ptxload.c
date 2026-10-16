// usage: ptxload FILE.ptx... : cuModuleLoadData + cuModuleGetFunction for every .entry
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int (*I)(unsigned); typedef int (*DG)(int*,int); typedef int (*CC)(void**,unsigned,int);
typedef int (*ML)(void**,const void*); typedef int (*GF)(void**,void*,const char*); typedef int (*MU)(void*);
int main(int argc,char**argv){
  void*l=dlopen("libcuda.so",RTLD_NOW); if(!l){fprintf(stderr,"%s\n",dlerror());return 1;}
  I init=dlsym(l,"cuInit"); DG dg=dlsym(l,"cuDeviceGet"); CC cc=dlsym(l,"cuCtxCreate_v2");
  ML ml=dlsym(l,"cuModuleLoadData"); GF gf=dlsym(l,"cuModuleGetFunction"); MU mu=dlsym(l,"cuModuleUnload");
  int d; void*ctx; init(0); dg(&d,0); cc(&ctx,0,d); int bad=0;
  for(int a=1;a<argc;a++){
    FILE*f=fopen(argv[a],"rb"); fseek(f,0,2); long n=ftell(f); rewind(f); char*s=malloc(n+1); fread(s,1,n,f); s[n]=0; fclose(f);
    void*m=0; int r=ml(&m,s); printf("%s load=%d",argv[a],r); if(r){bad++;puts("");free(s);continue;}
    for(char*p=strstr(s,".entry ");p;p=strstr(p+1,".entry ")){
      char name[256]; sscanf(p+7,"%255[A-Za-z0-9_$]",name); void*fn=0; int g=gf(&fn,m,name);
      printf(" %s=%d",name,g); if(g) bad++; }
    puts(""); mu(m); free(s);}
  return bad?1:0;}
