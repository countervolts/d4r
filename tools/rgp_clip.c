#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <stdio.h>
#include <string.h>
int main(){
  Display* d=XOpenDisplay(NULL); if(!d) return 1;
  Window w=XCreateSimpleWindow(d,DefaultRootWindow(d),0,0,1,1,0,0,0);
  Atom clip=XInternAtom(d,"CLIPBOARD",0), utf=XInternAtom(d,"UTF8_STRING",0), prop=XInternAtom(d,"D4RCLIP",0), incr=XInternAtom(d,"INCR",0);
  XSelectInput(d,w,PropertyChangeMask);
  XConvertSelection(d,clip,utf,prop,w,CurrentTime);
  XEvent e; do XNextEvent(d,&e); while(e.type!=SelectionNotify);
  if(e.xselection.property==None){fprintf(stderr,"no selection\n");return 2;}
  Atom type; int fmt; unsigned long n,after; unsigned char* data;
  XGetWindowProperty(d,w,prop,0,0x7fffffff,True,AnyPropertyType,&type,&fmt,&n,&after,&data);
  if(type==incr){ XFree(data);
    for(;;){ do XNextEvent(d,&e); while(!(e.type==PropertyNotify && e.xproperty.state==PropertyNewValue && e.xproperty.atom==prop));
      XGetWindowProperty(d,w,prop,0,0x7fffffff,True,AnyPropertyType,&type,&fmt,&n,&after,&data);
      if(n==0){XFree(data);break;} fwrite(data,1,n,stdout); XFree(data);} }
  else { fwrite(data,1,n,stdout); XFree(data);}
  return 0;
}
