// Copyright 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
// Pinned collision_primitive_core.py sphere/capsule/box contact geometry.
#include "mjwarp_box.cuh"
namespace flash::mjwarp {
namespace {
constexpr float minval=1.e-15f,maxval=1.e10f;
struct BoxManifold {int count;float dist[8];CV pos[8],normal[8];};
__device__ BoxManifold empty_box(int count) {
  BoxManifold out{};out.count=count;for (int k=0;k<8;++k) out.dist[k]=maxval;return out;
}
__device__ float safe_div(float x,float y) {return x/(y!=0 ? y : minval);}
__device__ float clamp(float v,float lo,float hi) {return fmaxf(lo,fminf(hi,v));}
__device__ void sphere_box(BoxManifold& out,int id,CV sphere,float radius,CV center,const float* R,CV size) {
  CV local=crotT(R,sphere-center),clamped;
  for (int k=0;k<3;++k) clamped[k]=clamp(local[k],-size[k],size[k]);
  CV direction=clamped-local;float dist=sqrtf(cdot(direction,direction));
  direction=dist==0 ? direction : direction*(1.0f/dist);
  CV pos,normal;
  if (dist<=minval) {
    float closest=2*(size[0]+size[1]+size[2]);int face=0;
    for (int i=0;i<6;++i) {
      float d=fabsf((i%2 ? 1.0f : -1.0f)*size[i/2]-local[i/2]);
      if (closest>d) {closest=d;face=i;}
    }
    CV nearest=cv(0,0,0);nearest[face/2]=face%2 ? -1.0f : 1.0f;
    pos=local+nearest*(radius-closest)*.5f;normal=crot(R,nearest);out.dist[id]=-closest-radius;
  } else {
    CV deepest=local+direction*radius;pos=.5f*(clamped+deepest);normal=crot(R,direction);out.dist[id]=dist-radius;
  }
  out.pos[id]=center+crot(R,pos);out.normal[id]=normal;
}
__device__ BoxManifold capsule_box(CV center,CV axis,float radius,float half_length,CV box_center,const float* R,CV size) {
  BoxManifold out=empty_box(2);CV pos=crotT(R,center-box_center);axis=crotT(R,axis);CV halfaxis=axis*half_length;
  int axisdir=int(halfaxis[0]>0)+2*int(halfaxis[1]>0)+4*int(halfaxis[2]>0);
  float bestdist=1.e32f,bestsegment=-12,bestbox=0;int type=-4,face=-12,corner=-123,edge=-123;
  for (int i=-1;i<=1;i+=2) {
    CV tip=pos+float(i)*halfaxis,point=tip;int outside=0,ax=-1;
    for (int j=0;j<3;++j) {
      if (point[j]<-size[j]) {++outside;ax=j;point[j]=-size[j];}
      else if (point[j]>size[j]) {++outside;ax=j;point[j]=size[j];}
    }
    if (outside>1) continue;
    CV dif=point-tip;float dist=cdot(dif,dif);
    if (dist<bestdist) {bestdist=dist;bestsegment=float(i);type=-2+i;face=ax;}
  }
  // Preserve pinned corner-major order. CPU2.3.7 uses edge-major order.
  for (int i=0;i<8;++i) for (int j=0;j<3;++j) {
    if (i&(1<<j)) continue;
    CV point=cv((i&1 ? 1.0f : -1.0f)*size[0],(i&2 ? 1.0f : -1.0f)*size[1],(i&4 ? 1.0f : -1.0f)*size[2]);point[j]=0;
    CV dif=point-pos;float u=-size[j]*dif[j],v=cdot(halfaxis,dif),ma=size[j]*size[j],mb=-size[j]*halfaxis[j],mc=half_length*half_length;
    float determinant=ma*mc-mb*mb;if (fabsf(determinant)<minval) continue;
    float inv=1.0f/determinant,x1=(mc*u-mb*v)*inv,x2=(ma*v-mb*u)*inv;int s1=1,s2=1;
    if (x1>1) {x1=1;s1=2;x2=safe_div(v-mb,mc);}
    else if (x1<-1) {x1=-1;s1=0;x2=safe_div(v+mb,mc);}
    bool over=x2>1;
    if (over || x2<-1) {
      if (over) {x2=1;s2=2;x1=safe_div(u-mb,ma);} else {x2=-1;s2=0;x1=safe_div(u+mb,ma);}
      if (x1>1) {x1=1;s1=2;} else if (x1<-1) {x1=-1;s1=0;}
    }
    dif=dif-halfaxis*x2;dif[j]+=size[j]*x1;float distance=cdot(dif,dif);int config=s1*3+s2;
    if (distance<bestdist-minval) {bestdist=distance;bestsegment=x2;bestbox=x1;corner=i+(1<<j)*(config/6);edge=j;type=config;}
  }
  if (type==-4) return out;
  // The pinned function also computes an unused 2D projected corner guess.
  // Every live use overwrites it with axisdir ^ corner; omit those dead values.
  float second=-4;
  if (type>=0 && type/3!=1) {
    int config=axisdir^corner;
    if (config!=0 && config!=7) {
      int mul;
      if (config==1 || config==2 || config==4) mul=1;else {mul=-1;config=7-config;}
      int ax=config==1 ? 0 : config==2 ? 1 : 2,ax1=(ax+1)%3,ax2=(ax+2)%3;
      if (axis[ax]*axis[ax]>.5f) {
        float m=2*safe_div(size[ax],fabsf(halfaxis[ax]));second=fminf(1-float(mul)*bestsegment,m);
      } else {
        float m=2*fminf(safe_div(size[ax1],fabsf(halfaxis[ax1])),safe_div(size[ax2],fabsf(halfaxis[ax2])));second=-fminf(1+float(mul)*bestsegment,m);
      }
      second*=float(mul);
    }
  } else if (type>=0 && type/3==1) {
    int config=(axisdir^corner)&(7-(1<<edge));
    if (config==1 || config==2 || config==4) {
      int ax=edge,ax1=(edge+1)%3,ax2=(edge+2)%3;
      if (fabsf(axis[ax1])>fabsf(axis[ax2])) ax1=ax2;ax2=3-ax-ax1;
      int mul;if (config&(1<<ax2)) {mul=1;second=1-bestsegment;} else {mul=-1;second=1+bestsegment;}
      float e1=2*safe_div(size[ax2],fabsf(halfaxis[ax2]));second=fminf(e1,second);
      float e2=bool(axisdir&(1<<ax))==bool(config&(1<<ax2)) ? 1-bestbox : 1+bestbox;
      e1=size[ax]*safe_div(e2,fabsf(halfaxis[ax]));second=fminf(e1,second);second*=float(mul);
    }
  } else if (type<0 && face!=-1) {
    int mul=type==-3 ? 1 : -1;second=2;CV tmp=pos-halfaxis*float(mul);
    for (int i=0;i<3;++i) if (i!=face) {
      float inv=safe_div(float(mul),halfaxis[i]);float e1=(size[i]-tmp[i])*inv;
      if (0<e1 && e1<second) second=e1;e1=(-size[i]-tmp[i])*inv;if (0<e1 && e1<second) second=e1;
    }
    second*=float(mul);
  }
  sphere_box(out,0,crot(R,pos+halfaxis*bestsegment)+box_center,radius,box_center,R,size);
  if (second>-3) sphere_box(out,1,crot(R,pos+halfaxis*(second+bestsegment))+box_center,radius,box_center,R,size);
  return out;
}
__device__ int clip_face(int count,CV* poly,int coordinate,float sign,float limit) {
  float d[8];bool inside=true;
  for (int i=0;i<count;++i) {d[i]=sign*poly[i][coordinate]-limit;if (!(d[i]<=0)) inside=false;}
  if (inside) return count;
  CV clipped[8]{};int n=0;
  for (int i=0;i<count;++i) {
    int next=i+1==count ? 0 : i+1;float dp=d[i],dq=d[next];
    if (dp<=0 && n<8) clipped[n++]=poly[i];
    if (((dp<0 && dq>0) || (dp>0 && dq<0)) && n<8) {float t=dp/(dp-dq);clipped[n++]=poly[i]+t*(poly[next]-poly[i]);}
  }
  for (int i=0;i<8;++i) poly[i]=clipped[i];return n;
}
__device__ BoxManifold box_box(CV p1,const float* R1,CV size1,CV p2,const float* R2,CV size2,float margin) {
  BoxManifold out=empty_box(0);CV pos21=crotT(R1,p2-p1),pos12=crotT(R2,p1-p2);float rot[9],absolute[9];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {rot[3*i+j]=cdot(ccolumn(R1,i),ccolumn(R2,j));absolute[3*i+j]=fabsf(rot[3*i+j]);}
  float septol=margin+1.e-6f*(size1[0]+size1[1]+size1[2]+size2[0]+size2[1]+size2[2]),best=-maxval;int code=-1;
  for (int i=0;i<3;++i) {
    float radius=absolute[3*i]*size2[0]+absolute[3*i+1]*size2[1]+absolute[3*i+2]*size2[2];float sep=fabsf(pos21[i])-size1[i]-radius;
    if (sep>septol) return out;if (sep>best) {best=sep;code=i;}
  }
  for (int j=0;j<3;++j) {
    float radius=absolute[j]*size1[0]+absolute[3+j]*size1[1]+absolute[6+j]*size1[2];float sep=fabsf(pos12[j])-size2[j]-radius;
    if (sep>septol) return out;if (sep>best) {best=sep;code=3+j;}
  }
  float face_best=best;int face_code=code;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    int i1=(i+1)%3,i2=(i+2)%3;float x=-rot[3*i2+j],y=rot[3*i1+j],norm2=x*x+y*y;
    if (norm2<1.e-7f) continue;float inv=1.0f/sqrtf(norm2);x*=inv;y*=inv;
    float radius1=size1[i1]*fabsf(x)+size1[i2]*fabsf(y);int j1=(j+1)%3,j2=(j+2)%3;
    float a1=x*rot[3*i1+j1]+y*rot[3*i2+j1],a2=x*rot[3*i1+j2]+y*rot[3*i2+j2];float radius2=size2[j1]*fabsf(a1)+size2[j2]*fabsf(a2);
    float sep=fabsf(x*pos21[i1]+y*pos21[i2])-radius1-radius2;if (sep>septol) return out;
    if (sep-1.e-6f*fabsf(sep)>best && sep>face_best) {best=sep;code=6+3*i+j;}
  }
  if (code<0) return out;
  int ei=0,ej=0,i1=0,i2=0;CV axis=cv(0,0,0);
  if (code>=6) {
    ei=(code-6)/3;ej=(code-6)%3;i1=(ei+1)%3;i2=(ei+2)%3;axis[i1]=-rot[3*i2+ej];axis[i2]=rot[3*i1+ej];axis=cnormalize(axis);
    float alignment=face_code<3 ? fabsf(axis[face_code]) : fabsf(cdot(axis,ccolumn(rot,face_code-3)));
    if (alignment>.99f && best<face_best+.05f*fabsf(face_best)+minval) {code=face_code;best=face_best;}
  }
  if (code>=6) {
    int j1=(ej+1)%3,j2=(ej+2)%3;if (cdot(axis,pos21)<0) axis=-axis;CV a2=crotT(rot,axis);int amb1=-1,amb2=-1;
    if (fabsf(axis[i1])<1.e-5f) amb1=i1;else if (fabsf(axis[i2])<1.e-5f) amb1=i2;
    if (fabsf(a2[j1])<1.e-5f) amb2=j1;else if (fabsf(a2[j2])<1.e-5f) amb2=j2;
    CV d2=ccolumn(rot,ej);float b=d2[ei],denom=1-b*b,best_dist=maxval;CV w1=cv(0,0,0),w2=cv(0,0,0);
    for (int v1=0;v1<(amb1>=0 ? 2 : 1);++v1) for (int v2=0;v2<(amb2>=0 ? 2 : 1);++v2) {
      CV c1=cv(0,0,0);c1[i1]=axis[i1]>=0 ? size1[i1] : -size1[i1];c1[i2]=axis[i2]>=0 ? size1[i2] : -size1[i2];if (amb1>=0 && v1) c1[amb1]=-c1[amb1];
      CV corner=cv(0,0,0);corner[j1]=a2[j1]>=0 ? -size2[j1] : size2[j1];corner[j2]=a2[j2]>=0 ? -size2[j2] : size2[j2];if (amb2>=0 && v2) corner[amb2]=-corner[amb2];
      CV c2=crot(rot,corner)+pos21,e=c2-c1;float d1e=e[ei],d2e=cdot(d2,e),s=denom>=minval ? (d1e-b*d2e)/denom : 0;
      s=clamp(s,-size1[ei],size1[ei]);float t=clamp(b*s-d2e,-size2[ej],size2[ej]);s=clamp(d1e+b*t,-size1[ei],size1[ei]);
      CV q1=c1;q1[ei]=s;CV q2=c2+d2*t,gap=q2-q1;float distance=cdot(gap,gap);if (distance<best_dist) {best_dist=distance;w1=q1;w2=q2;}
    }
    float dist=cdot(w2-w1,axis);if (dist>septol) return out;
    out.count=1;out.dist[0]=dist;out.pos[0]=crot(R1,.5f*(w1+w2))+p1;out.normal[0]=crot(R1,axis);return out;
  }
  bool ref1=code<3;int a=ref1 ? code : code-3;CV size_ref=ref1 ? size1 : size2,size_inc=ref1 ? size2 : size1,pos_ref=ref1 ? p1 : p2,posoi=ref1 ? pos21 : pos12;const float* matref=ref1 ? R1 : R2;float rinc[9];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) rinc[3*i+j]=ref1 ? rot[3*i+j] : rot[3*j+i];
  float sign=posoi[a]>=0 ? 1.0f : -1.0f;int incident=0;
  for (int k=1;k<3;++k) if (fabsf(rinc[3*a+k])>fabsf(rinc[3*a+incident])) incident=k;
  float incsign=sign*rinc[3*a+incident]>0 ? -1.0f : 1.0f;int ax=(a+1)%3,ay=(a+2)%3,bu=(incident+1)%3,bv=(incident+2)%3,coords[3]={ax,ay,a};CV cx,du,dv;
  for (int r=0;r<3;++r) {int c=coords[r];cx[r]=posoi[c]+incsign*size_inc[incident]*rinc[3*c+incident];du[r]=size_inc[bu]*rinc[3*c+bu];dv[r]=size_inc[bv]*rinc[3*c+bv];}
  cx[2]=sign*cx[2]-size_ref[a];du[2]*=sign;dv[2]*=sign;CV poly[8]{};
  for (int k=0;k<4;++k) {float su=k==0 || k==3 ? 1.0f : -1.0f,sv=k<2 ? 1.0f : -1.0f;poly[k]=cx+su*du+sv*dv;}
  int count=4;count=clip_face(count,poly,0,1,size_ref[ax]);count=clip_face(count,poly,0,-1,size_ref[ax]);count=clip_face(count,poly,1,1,size_ref[ay]);count=clip_face(count,poly,1,-1,size_ref[ay]);
  int accepted=0;float duplicate=1.e-10f*(size_ref[ax]*size_ref[ax]+size_ref[ay]*size_ref[ay]);
  for (int k=0;k<count;++k) {
    if (poly[k][2]>margin) continue;bool dupe=false;
    for (int q=0;q<accepted;++q) {float dx=out.pos[q][0]-poly[k][0],dy=out.pos[q][1]-poly[k][1];if (dx*dx+dy*dy<duplicate) {dupe=true;break;}}
    if (!dupe) out.pos[accepted++]=poly[k];
  }
  if (!accepted) return out;CV normal=(ref1 ? sign : -sign)*ccolumn(matref,a);
  for (int k=0;k<accepted;++k) {CV v=out.pos[k],local=cv(0,0,0);local[ax]=v[0];local[ay]=v[1];local[a]=sign*(size_ref[a]+.5f*v[2]);out.dist[k]=v[2];out.pos[k]=crot(matref,local)+pos_ref;out.normal[k]=normal;}
  out.count=accepted;return out;
}
__global__ void box_contacts(ContactInputs a,int boxbox_enabled) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x;i<a.capacity;i+=gridDim.x*blockDim.x) {
    if (i>=a.candidate_count[0]) continue;int g1=a.candidate_geom[2*i],g2=a.candidate_geom[2*i+1],t1=a.geom_type[g1],t2=a.geom_type[g2];
    if (t2!=6 || (t1!=2 && t1!=3 && !(t1==6 && boxbox_enabled))) continue;
    int w=a.candidate_world[i],pair=a.candidate_ids[2*i],sensor=a.candidate_ids[2*i+1];ContactMaterial p=contact_material(a,w,g1,g2,pair);
    CV p1=cload(a.xpos,w*a.geoms+g1),p2=cload(a.xpos,w*a.geoms+g2),size1=cload(contact_parameter(a.size,w,a.geoms,g1,3)),size2=cload(contact_parameter(a.size,w,a.geoms,g2,3));
    const float *R1=a.xmat+9*(w*a.geoms+g1),*R2=a.xmat+9*(w*a.geoms+g2);BoxManifold out=empty_box(1);
    if (t1==2) sphere_box(out,0,p1,size1[0],p2,R2,size2);
    else if (t1==3) out=capsule_box(p1,ccolumn(R1,2),size1[0],size1[1],p2,R2,size2);
    else out=box_box(p1,R1,size1,p2,R2,size2,p.margin+p.gap);
    for (int k=0;k<out.count;++k) {CV frame[3];contact_frame(out.normal[k],frame);publish_contact(a,p,w,g1,g2,pair,sensor,k,out.dist[k],out.pos[k],frame);}
  }
}
}
}
extern "C" {
const char* flash_mjwarp_box_source_id() {return FLASH_MJW_BOX_SOURCE_ID;}
std::size_t flash_mjwarp_box_arguments_bytes() {return sizeof(flash::mjwarp::ContactInputs);}
int flash_mjwarp_box_arch() {return FLASH_MJW_BOX_ARCH;}
int flash_mjwarp_box_launch(const flash::mjwarp::ContactInputs* a,int boxbox_enabled,cudaStream_t stream) {
  if (!a || a->worlds<1 || a->geoms<1 || a->capacity<0 || (boxbox_enabled!=0 && boxbox_enabled!=1)) return int(cudaErrorInvalidValue);
  if (a->capacity) {int blocks=(a->capacity+127)/128;if (blocks>4096) blocks=4096;flash::mjwarp::box_contacts<<<blocks,128,0,stream>>>(*a,boxbox_enabled);}
  return int(cudaGetLastError());
}
}
