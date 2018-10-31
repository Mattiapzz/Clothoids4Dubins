/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2017                                                      |
 |                                                                          |
 |         , __                 , __                                        |
 |        /|/  \               /|/  \                                       |
 |         | __/ _   ,_         | __/ _   ,_                                |
 |         |   \|/  /  |  |   | |   \|/  /  |  |   |                        |
 |         |(__/|__/   |_/ \_/|/|(__/|__/   |_/ \_/|/                       |
 |                           /|                   /|                        |
 |                           \|                   \|                        |
 |                                                                          |
 |      Enrico Bertolazzi                                                   |
 |      Dipartimento di Ingegneria Industriale                              |
 |      Universita` degli Studi di Trento                                   |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Line.hh"
#include "Circle.hh"
#include "Biarc.hh"
#include "Clothoid.hh"
#include "CubicRootsFlocke.hh"

#include <cmath>
#include <cfloat>

// workaround for windows that defines max and min as macros!
#ifdef max
  #undef max
#endif
#ifdef min
  #undef min
#endif

namespace G2lib {

  using std::vector;
  using std::abs;
  using std::min;
  using std::max;
  using std::swap;
  using std::ceil;
  using std::floor;
  using std::isfinite;
  using std::numeric_limits;

  int_type  ClothoidCurve::max_iter  = 10;
  real_type ClothoidCurve::tolerance = 1e-9;

  // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

  void
  ClothoidCurve::optimized_sample_internal(
    real_type           s_begin,
    real_type           s_end,
    real_type           offs,
    real_type           ds,
    real_type           max_angle,
    vector<real_type> & s
  ) const {
    real_type ss  = s_begin;
    real_type thh = theta(s_begin);
    for ( int_type npts = 0; ss < s_end; ++npts ) {
      G2LIB_ASSERT( npts < 1000000,
                    "ClothoidCurve::optimized_sample_internal " <<
                    "is generating too much points (>1000000)\n" <<
                    "something is going wrong or parameters are not well set" );
      // estimate angle variation and compute step accodingly
      real_type k   = CD.kappa( ss );
      real_type dss = ds/(1-k*offs); // scale length with offset
      real_type sss = ss + dss;
      if ( sss > s_end ) {
        sss = s_end;
        dss = s_end-ss;
      }
      if ( abs(k*dss) > max_angle ) {
        dss = abs(max_angle/k);
        sss = ss + dss;
      }
      // check and recompute if necessary
      real_type thhh = theta(sss);
      if ( abs(thh-thhh) > max_angle ) {
        k    = CD.kappa( sss );
        dss  = abs(max_angle/k);
        sss  = ss + dss;
        thhh = theta(sss);
      }
      ss  = sss;
      thh = thhh;
      s.push_back(ss);
    }
    s.back() = s_end;
  }

  // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

  void
  ClothoidCurve::optimized_sample( real_type           offs,
                                   int_type            npts,
                                   real_type           max_angle,
                                   vector<real_type> & s ) const {
    s.clear();
    s.reserve( size_t(npts) );
    s.push_back(0);

    real_type ds = L/npts;
    if ( CD.kappa0*CD.dk >= 0 || CD.kappa(L)*CD.dk <= 0 ) {
      optimized_sample_internal( 0, L, offs, ds, max_angle, s );
    } else {
      // flex inside, split clothoid
      real_type sflex = -CD.kappa0/CD.dk;
      optimized_sample_internal( 0, sflex, offs, ds, max_angle, s );
      optimized_sample_internal( sflex, L, offs, ds, max_angle, s );
    }
  }

  // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

  /*\
   |  _    _   _____    _                _
   | | |__| |_|_   _| _(_)__ _ _ _  __ _| |___
   | | '_ \ '_ \| || '_| / _` | ' \/ _` | / -_)
   | |_.__/_.__/|_||_| |_\__,_|_||_\__, |_\___|
   |                               |___/
  \*/

  // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  void
  ClothoidCurve::bbTriangles_internal(
    real_type     offs,
    vector<T2D> & tvec,
    real_type     s_begin,
    real_type     s_end,
    real_type     max_angle,
    real_type     max_size
  ) const {

    static real_type const one_degree = m_pi/180;

    real_type ss  = s_begin;
    real_type thh = CD.theta(ss);
    real_type MX  = min( L, max_size );
    for ( int_type npts = 0; ss < s_end; ++npts ) {
      G2LIB_ASSERT( npts < 1000000,
                    "ClothoidCurve::bbTriangles_internal " <<
                    "is generating too much triangles (>1000000)\n" <<
                    "something is going wrong or parameters are not well set" );

      // estimate angle variation and compute step accodingly
      real_type k   = CD.kappa( ss );
      real_type dss = MX/(1-k*offs); // scale length with offset
      real_type sss = ss + dss;
      if ( sss > s_end ) {
        sss = s_end;
        dss = s_end-ss;
      }
      if ( abs(k*dss) > max_angle ) {
        dss = abs(max_angle/k);
        sss = ss + dss;
      }
      // check and recompute if necessary
      real_type thhh = theta(sss);
      if ( abs(thh-thhh) > max_angle ) {
        k    = CD.kappa( sss );
        dss  = abs(max_angle/k);
        sss  = ss + dss;
        thhh = theta(sss);
      }

      real_type x0, y0, x1, y1;
      CD.eval( ss,  offs, x0, y0 );
      CD.eval( sss, offs, x1, y1 );

      real_type tx0    = cos(thh);
      real_type ty0    = sin(thh);
      real_type alpha  = sss-ss; // se angolo troppo piccolo uso approx piu rozza
      if ( abs(thh-thhh) > one_degree ) {
        real_type tx1 = cos(thhh);
        real_type ty1 = sin(thhh);
        real_type det = tx1 * ty0 - tx0 * ty1;
        real_type dx  = x1-x0;
        real_type dy  = y1-y0;
        alpha = (dy*tx1 - dx*ty1)/det;
      }

      real_type x2 = x0 + alpha*tx0;
      real_type y2 = y0 + alpha*ty0;
      T2D t( x0, y0, x2, y2, x1, y1, ss, sss );

      tvec.push_back( t );

      ss  = sss;
      thh = thhh;
    }
  }

  // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

  void
  ClothoidCurve::bbTriangles( real_type     offs,
                              vector<T2D> & tvec,
                              real_type     max_angle,
                              real_type     max_size ) const {

    if ( CD.kappa0*CD.dk >= 0 || CD.kappa(L)*CD.dk <= 0 ) {
      bbTriangles_internal( offs, tvec, 0, L, max_angle, max_size );
    } else {
      // flex inside, split clothoid
      real_type sflex = -CD.kappa0/CD.dk;
      bbTriangles_internal( offs, tvec, 0, sflex, max_angle, max_size );
      bbTriangles_internal( offs, tvec, sflex, L, max_angle, max_size );
    }
  }

  // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  /*\
   |  ___ ___
   | | _ ) _ ) _____ __
   | | _ \ _ \/ _ \ \ /
   | |___/___/\___/_\_\
  \*/

  void
  ClothoidCurve::bbox( real_type   offs,
                       real_type & xmin,
                       real_type & ymin,
                       real_type & xmax,
                       real_type & ymax ) const {
    vector<T2D> tvec;
    bbTriangles( offs, tvec, m_pi/18, 1e100 );
    xmin = ymin = numeric_limits<real_type>::infinity();
    xmax = ymax = -xmin;
    vector<T2D>::const_iterator it;
    for ( it = tvec.begin(); it != tvec.end(); ++it ) {
      // - - - - - - - - - - - - - - - - - - - -
      if      ( it->x1() < xmin ) xmin = it->x1();
      else if ( it->x1() > xmax ) xmax = it->x1();
      if      ( it->x2() < xmin ) xmin = it->x2();
      else if ( it->x2() > xmax ) xmax = it->x2();
      if      ( it->x3() < xmin ) xmin = it->x3();
      else if ( it->x3() > xmax ) xmax = it->x3();
      // - - - - - - - - - - - - - - - - - - - -
      if      ( it->y1() < ymin ) ymin = it->y1();
      else if ( it->y1() > ymax ) ymax = it->y1();
      if      ( it->y2() < ymin ) ymin = it->y2();
      else if ( it->y2() > ymax ) ymax = it->y2();
      if      ( it->y3() < ymin ) ymin = it->y3();
      else if ( it->y3() > ymax ) ymax = it->y3();
    }
  }

  // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
  /*\
   |     _        _    ____  ____  _
   |    / \      / \  | __ )| __ )| |_ _ __ ___  ___
   |   / _ \    / _ \ |  _ \|  _ \| __| '__/ _ \/ _ \
   |  / ___ \  / ___ \| |_) | |_) | |_| | |  __/  __/
   | /_/   \_\/_/   \_\____/|____/ \__|_|  \___|\___|
  \*/
  void
  ClothoidCurve::build_AABBtree( real_type offs,
                                 real_type max_angle,
                                 real_type max_size ) const {
    if ( aabb_done &&
         isZero( offs-aabb_offs ) &&
         isZero( max_angle-aabb_max_angle ) &&
         isZero( max_size-aabb_max_size ) ) return;

    #ifdef G2LIB_USE_CXX11
    vector<shared_ptr<BBox const> > bboxes;
    #else
    vector<BBox const *> bboxes;
    #endif

    bbTriangles( offs, aabb_tri, max_angle, max_size );
    bboxes.reserve(aabb_tri.size());
    vector<T2D>::const_iterator it;
    int_type ipos = 0;
    for ( it = aabb_tri.begin(); it != aabb_tri.end(); ++it, ++ipos ) {
      real_type xmin, ymin, xmax, ymax;
      it->bbox( xmin, ymin, xmax, ymax );
      #ifdef G2LIB_USE_CXX11
      bboxes.push_back( make_shared<BBox const>(
        xmin, ymin, xmax, ymax, G2LIB_CLOTHOID, ipos
      ) );
      #else
      bboxes.push_back(
        new BBox( xmin, ymin, xmax, ymax, G2LIB_CLOTHOID, ipos )
      );
      #endif
    }
    aabb_tree.build(bboxes);
    aabb_done      = true;
    aabb_offs      = offs;
    aabb_max_angle = max_angle;
    aabb_max_size  = max_size;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  /*\
   |            _ _ _     _
   |   ___ ___ | | (_)___(_) ___  _ __
   |  / __/ _ \| | | / __| |/ _ \| '_ \
   | | (_| (_) | | | \__ \ | (_) | | | |
   |  \___\___/|_|_|_|___/_|\___/|_| |_|
  \*/
  bool
  ClothoidCurve::collision( BaseCurve const & obj ) const {
    bool ok = false;
    switch ( obj.type() ) {
    case G2LIB_LINE:
      { // promote
        ClothoidCurve C(*static_cast<LineSegment const*>(&obj));
        ok = this->collision( C );
      }
      break;
    case G2LIB_CIRCLE:
      {
        ClothoidCurve C(*static_cast<CircleArc const*>(&obj));
        ok = this->collision( C );
      }
      break;
    case G2LIB_BIARC:
      {
        Biarc B(*static_cast<Biarc const*>(&obj));
        ok = B.collision( *this );
      }
      break;
    case G2LIB_CLOTHOID:
      {
        ClothoidCurve const & C = *static_cast<ClothoidCurve const*>(&obj);
        ok = this->collision( C );
      }
      break;
    case G2LIB_POLYLINE:
    case G2LIB_CLOTHOID_LIST:
      G2LIB_ASSERT( false, "ClothoidCurve::collision!" );
    }
    return ok;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  bool
  ClothoidCurve::collision( real_type         offs,
                            BaseCurve const & obj,
                            real_type         offs_obj ) const {
    bool ok = false;
    switch ( obj.type() ) {
    case G2LIB_LINE:
      { // promote
        ClothoidCurve C(*static_cast<LineSegment const*>(&obj));
        ok = this->collision( offs, C, offs_obj );
      }
      break;
    case G2LIB_CIRCLE:
      {
        ClothoidCurve C(*static_cast<CircleArc const*>(&obj));
        ok = this->collision( offs, C, offs_obj );
      }
      break;
    case G2LIB_BIARC:
      {
        Biarc B(*static_cast<Biarc const*>(&obj));
        ok = B.collision( offs_obj, *this, offs );
      }
      break;
    case G2LIB_CLOTHOID:
      {
        ClothoidCurve const & C = *static_cast<ClothoidCurve const*>(&obj);
        ok = this->collision( offs, C, offs_obj );
      }
      break;
    case G2LIB_POLYLINE:
    case G2LIB_CLOTHOID_LIST:
      G2LIB_ASSERT( false, "ClothoidCurve::collision!" );
    }
    return ok;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  bool
  ClothoidCurve::collision( ClothoidCurve const & C ) const {
    G2LIB_ASSERT( false, "DA FARE ClothoidCurve::collision" );
    return false;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  bool
  ClothoidCurve::collision( real_type             offs,
                            ClothoidCurve const & C,
                            real_type             offs_C ) const {
    G2LIB_ASSERT( false, "DA FARE ClothoidCurve::collision" );
    return false;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // collision detection
  bool
  ClothoidCurve::approximate_collision(
    real_type             offs,
    ClothoidCurve const & clot,
    real_type             clot_offs,
    real_type             max_angle,
    real_type             max_size
  ) const {

  /*
    vector<bbData> bbV0, bbV1;
    bbSplit( max_angle, max_size, offs, bbV0 );
    clot.bbSplit( max_angle, max_size, clot_offs, bbV1 );
    for ( unsigned i = 0; i < unsigned(bbV0.size()); ++i ) {
      bbData & bbi = bbV0[i];
      for ( unsigned j = 0; j < unsigned(bbV1.size()); ++j ) {
        bbData & bbj = bbV1[j];
        if ( bbi.t.overlap(bbj.t) ) return true;
      }
    }
    */
    return false;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  /*\
   |  _       _                          _
   | (_)_ __ | |_ ___ _ __ ___  ___  ___| |_
   | | | '_ \| __/ _ \ '__/ __|/ _ \/ __| __|
   | | | | | | ||  __/ |  \__ \  __/ (__| |_
   | |_|_| |_|\__\___|_|  |___/\___|\___|\__|
  \*/

  void
  ClothoidCurve::intersect( BaseCurve const & obj,
                            IntersectList   & ilist,
                            bool              swap_s_vals ) const {
    switch ( obj.type() ) {
    case G2LIB_LINE:
      { // promote to clothoid
        ClothoidCurve C(*static_cast<LineSegment const*>(&obj));
        this->intersect( C, ilist, swap_s_vals );
      }
      break;
    case G2LIB_CIRCLE:
      {
        ClothoidCurve C(*static_cast<CircleArc const*>(&obj));
        this->intersect( C, ilist, swap_s_vals );
      }
      break;
    case G2LIB_BIARC:
      {
        Biarc B(*static_cast<Biarc const*>(&obj));
        B.intersect( *this, ilist, !swap_s_vals );
      }
      break;
    case G2LIB_CLOTHOID:
      {
        ClothoidCurve const & C = *static_cast<ClothoidCurve const*>(&obj);
        this->intersect( C, ilist, swap_s_vals );
      }
      break;
    case G2LIB_POLYLINE:
    case G2LIB_CLOTHOID_LIST:
      G2LIB_ASSERT( false, "CircleArc::intersect!" );
    }
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  void
  ClothoidCurve::intersect( real_type         offs,
                            BaseCurve const & obj,
                            real_type         offs_obj,
                            IntersectList   & ilist,
                            bool              swap_s_vals ) const {
    switch ( obj.type() ) {
    case G2LIB_LINE:
      { // promote to clothoid
        ClothoidCurve C(*static_cast<LineSegment const*>(&obj));
        this->intersect( offs, C, offs_obj, ilist, swap_s_vals );
      }
      break;
    case G2LIB_CIRCLE:
      {
        ClothoidCurve C(*static_cast<CircleArc const*>(&obj));
        this->intersect( offs, C, offs_obj, ilist, swap_s_vals );
      }
      break;
    case G2LIB_BIARC:
      {
        Biarc B(*static_cast<Biarc const*>(&obj));
        B.intersect( offs_obj, *this, offs, ilist, !swap_s_vals );
      }
      break;
    case G2LIB_CLOTHOID:
      {
        ClothoidCurve const & C = *static_cast<ClothoidCurve const*>(&obj);
        this->intersect( offs, C, offs_obj, ilist, swap_s_vals );
      }
      break;
    case G2LIB_POLYLINE:
    case G2LIB_CLOTHOID_LIST:
      G2LIB_ASSERT( false, "CircleArc::intersect!" );
    }
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  void
  ClothoidCurve::intersect( ClothoidCurve const & C,
                            IntersectList       & ilist,
                            bool                  swap_s_vals ) const {
#if 0
    vector<real_type> s1, s2;
    this->intersect( C, s1, s2,
                     ClothoidCurve::max_iter,
                     ClothoidCurve::tolerance );
    ilist.reserve( ilist.size() + s1.size() );
    for ( size_t i = 0; i < s1.size(); ++i ) {
      real_type ss1 = s1[i];
      real_type ss2 = s2[i];
      if ( swap_s_vals ) swap( ss1, ss2 );
      ilist.push_back( Ipair( ss1, ss2 ) );
    }
#else
  intersect( 0, C, 0, ilist, swap_s_vals );
#endif
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  void
  ClothoidCurve::intersect( real_type             offs,
                            ClothoidCurve const & C,
                            real_type             offs_C,
                            IntersectList       & ilist,
                            bool                  swap_s_vals ) const {
#if 1
    this->build_AABBtree( offs );
    C.build_AABBtree( offs_C );
    AABBtree::VecPairPtrBBox intersectionList;
    aabb_tree.intersect( C.aabb_tree, intersectionList );
    AABBtree::VecPairPtrBBox::const_iterator ip;

    for ( ip = intersectionList.begin(); ip != intersectionList.end(); ++ip ) {
      size_t ipos1 = size_t(ip->first->Ipos());
      size_t ipos2 = size_t(ip->second->Ipos());

      T2D const & T1 = aabb_tri[ipos1];
      T2D const & T2 = C.aabb_tri[ipos2];

      real_type s1_min = T1.s0;
      real_type s1_max = T1.s1;
      real_type ss1    = (s1_min+s1_max)/2;
      real_type s2_min = T2.s0;
      real_type s2_max = T2.s1;
      real_type ss2    = (s2_min+s2_max)/2;
      int_type  nout   = 0;
      bool converged = false;

      for ( int_type i = 0; i < max_iter && !converged; ++i ) {
        real_type t1[2], t2[2], p1[2], p2[2];
        CD.eval  ( ss1, offs, p1[0], p1[1] );
        CD.eval_D( ss1, offs, t1[0], t1[1] );
        C.CD.eval  ( ss2, offs_C, p2[0], p2[1] );
        C.CD.eval_D( ss2, offs_C, t2[0], t2[1] );
        /*
        // risolvo il sistema
        // p1 + alpha * t1 = p2 + beta * t2
        // alpha * t1 - beta * t2 = p2 - p1
        //
        //  / t1[0] -t2[0] \ / alpha \ = / p2[0] - p1[0] \
        //  \ t1[1] -t2[1] / \ beta  /   \ p2[1] - p1[1] /
        */
        real_type det = t2[0]*t1[1]-t1[0]*t2[1];
        real_type px  = p2[0]-p1[0];
        real_type py  = p2[1]-p1[1];
        ss1 += (py*t2[0] - px*t2[1])/det;
        ss2 += (t1[0]*py - t1[1]*px)/det;
        if ( ! ( isfinite(ss1) && isfinite(ss1) ) ) break;
        bool out = false;
        if      ( ss1 <= s1_min ) { out = true; ss1 = s1_min; }
        else if ( ss1 >= s1_max ) { out = true; ss1 = s1_max; }
        if      ( ss2 <= s2_min ) { out = true; ss2 = s2_min; }
        else if ( ss2 >= s2_max ) { out = true; ss2 = s2_max; }
        if ( out ) {
          if ( ++nout > 3 ) break;
        } else {
          converged = abs(px) <= tolerance && abs(py) <= tolerance;
        }
      }

      if ( converged ) {
        if ( swap_s_vals ) swap( ss1, ss2 );
        ilist.push_back( Ipair( ss1, ss2 ) );
      }
    }

#else
    vector<real_type> s1, s2;
    this->intersect( offs, C, offs_C, s1, s2,
                     ClothoidCurve::max_iter,
                     ClothoidCurve::tolerance );
    ilist.reserve( ilist.size() + s1.size() );
    for ( size_t i = 0; i < s1.size(); ++i ) {
      real_type ss1 = s1[i];
      real_type ss2 = s2[i];
      if ( swap_s_vals ) swap( ss1, ss2 );
      ilist.push_back( Ipair( ss1, ss2 ) );
    }
#endif
  }

  /*\
   |                  _           _   _
   |  _ __  _ __ ___ (_) ___  ___| |_(_) ___  _ __
   | | '_ \| '__/ _ \| |/ _ \/ __| __| |/ _ \| '_ \
   | | |_) | | | (_) | |  __/ (__| |_| | (_) | | | |
   | | .__/|_|  \___// |\___|\___|\__|_|\___/|_| |_|
   | |_|           |__/
  \*/
  int_type
  ClothoidCurve::projection( real_type   qx,
                             real_type   qy,
                             real_type & x,
                             real_type & y,
                             real_type & s ) const {
    G2LIB_ASSERT( false, "DA FARE ClothoidCurve::projection" );
    return 0;
  }

  int_type // true if projection is unique and orthogonal
  ClothoidCurve::projection( real_type   qx,
                             real_type   qy,
                             real_type   offs,
                             real_type & x,
                             real_type & y,
                             real_type & s ) const {
    G2LIB_ASSERT( false, "DA FARE ClothoidCurve::projection" );
    return 0;
  }

  real_type
  ClothoidCurve::closestPoint( real_type   qx,
                               real_type   qy,
                               real_type   offs,
                               real_type & x,
                               real_type & y,
                               real_type & s ) const {
    G2LIB_ASSERT( false, "DA FARE ClothoidCurve::closestPoint" );
    return 0;
  }


  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  real_type
  ClothoidCurve::thetaTotalVariation() const {
    // cerco punto minimo parabola
    // root = -k/dk;
    real_type kL  = CD.kappa0;
    real_type kR  = CD.kappa(L);
    real_type thL = 0;
    real_type thR = CD.deltaTheta(L);
    if ( kL*kR < 0 ) {
      real_type root = -CD.kappa0/CD.dk;
      if ( root > 0 && root < L ) {
        real_type thM  = CD.deltaTheta(root);
        return abs( thR - thM ) + abs( thM - thL );
      }
    }
    return abs( thR - thL );
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  real_type
  ClothoidCurve::thetaMinMax( real_type & thMin, real_type & thMax ) const {
    // cerco punto minimo parabola
    // root = -k/dk;
    real_type kL  = CD.kappa0;
    real_type kR  = CD.kappa(L);
    real_type thL = 0;
    real_type thR = CD.deltaTheta(L);
    if ( thL < thR ) { thMin = thL; thMax = thR; }
    else             { thMin = thR; thMax = thL; }
    if ( kL*kR < 0 ) {
      real_type root = -CD.kappa0/CD.dk;
      if ( root > 0 && root < L ) {
        real_type thM = CD.deltaTheta(root);
        if      ( thM < thMin ) thMin = thM;
        else if ( thM > thMax ) thMax = thM;
      }
    }
    return thMax - thMin;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  real_type
  ClothoidCurve::curvatureMinMax( real_type & kMin, real_type & kMax ) const {
    // cerco punto minimo parabola
    // root = -k/dk;
    kMin = CD.kappa0;
    kMax = CD.kappa(L);
    if ( kMax < kMin ) swap( kMax, kMin );
    return kMax - kMin;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  real_type
  ClothoidCurve::curvatureTotalVariation() const {
    // cerco punto minimo parabola
    // root = -k/dk;
    real_type km = CD.kappa0;
    real_type kp = CD.kappa(L);
    return abs(kp-km);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  real_type
  ClothoidCurve::integralCurvature2() const {
    return L*( CD.kappa0*(CD.kappa0+L*CD.dk) + (L*L)*CD.dk*CD.dk/3 );
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  real_type
  ClothoidCurve::integralJerk2() const {
    real_type k2 = CD.kappa0*CD.kappa0;
    real_type k3 = CD.kappa0*k2;
    real_type k4 = k2*k2;
    real_type t1 = L;
    real_type t2 = L*t1;
    real_type t3 = L*t2;
    real_type t4 = L*t3;
    return ((((t4/5*CD.dk+t3*CD.kappa0)*CD.dk+(1+2*t2)*k2)*CD.dk+2*t1*k3)*CD.dk+k4)*L;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  real_type
  ClothoidCurve::integralSnap2() const {
    real_type k2  = CD.kappa0*CD.kappa0;
    real_type k3  = CD.kappa0*k2;
    real_type k4  = k2*k2;
    real_type k5  = k4*CD.kappa0;
    real_type k6  = k4*k2;
    real_type dk2 = CD.dk*CD.dk;
    real_type dk3 = CD.dk*dk2;
    real_type dk4 = dk2*dk2;
    real_type dk5 = dk4*CD.dk;
    real_type dk6 = dk4*dk2;
    real_type t2  = L;
    real_type t3  = L*t2;
    real_type t4  = L*t3;
    real_type t5  = L*t4;
    real_type t6  = L*t5;
    real_type t7  = L*t6;

    return ( (t7/7)*dk6 + dk5*CD.kappa0*t6 + 3*dk4*k2*t5 + 5*dk3*k3*t4 +
             5*dk2*k4*t3 + 3*dk3*t3 + 3*CD.dk*k5*t2 + 9*dk2*CD.kappa0*t2 +
             k6+9*k2*CD.dk ) * L;
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  bool
  ClothoidCurve::findST( real_type   x,
                         real_type   y,
                         real_type & s,
                         real_type & t ) const {
    real_type X, Y, nx, ny;
    real_type d = closestPoint( x, y, X, Y, s );
    nor( s, nx, ny );
    t = nx*(x-X) + ny*(y-Y);
    // check if projection is orthogonal on the curve
    #if 0
      real_type abst = abs(t);
      return abs(d-abst) <= machepsi1000*(1+abst);
    #else
      eval( s, t, X, Y );
      real_type err = hypot( x-X, y-Y );
      return err < 1e-8*(1+d);
    #endif
    //return abs(d-abst) <= 1e-3*(1+abst);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  ostream_type &
  operator << ( ostream_type & stream, ClothoidCurve const & c ) {
    stream <<   "x0     = " << c.CD.x0
           << "\ny0     = " << c.CD.y0
           << "\ntheta0 = " << c.CD.theta0
           << "\nkappa0 = " << c.CD.kappa0
           << "\ndk     = " << c.CD.dk
           << "\nL      = " << c.L
           << "\n";
    return stream;
  }

}

// EOF: Clothoid.cc
