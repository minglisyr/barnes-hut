use crate::{body::Body, quadtree::Quadtree, utils};

use broccoli::aabb::Rect;
use broccoli_rayon::{build::RayonBuildPar, prelude::RayonQueryPar};
use ultraviolet::Vec2;

pub struct Simulation {
    pub dt: f32,
    pub frame: usize,
    pub bodies: Vec<Body>,
    pub quadtree: Quadtree,
}

impl Simulation {
    pub fn new() -> Self {
        let dt = 0.05;
        let n = 200000;
        let theta = 1.0;
        let epsilon = 1.0;
        let leaf_capacity = 16;
        let thread_capacity = 1024;

        let bodies: Vec<Body> = utils::uniform_disc(n);
        let quadtree = Quadtree::new(theta, epsilon, leaf_capacity, thread_capacity);

        Self {
            dt,
            frame: 0,
            bodies,
            quadtree,
        }
    }

    pub fn step(&mut self) {
        self.iterate();
        self.collide();
        self.attract();
        self.update_colors();
        self.frame += 1;
    }

    pub fn attract(&mut self) {
        self.quadtree.build(&mut self.bodies);
        self.quadtree.acc(&mut self.bodies);
    }

    pub fn iterate(&mut self) {
        for body in &mut self.bodies {
            body.update(self.dt);
        }
    }

    pub fn collide(&mut self) {
        let mut rects = self
            .bodies
            .iter()
            .enumerate()
            .map(|(index, body)| {
                let pos = body.pos;
                let radius = body.radius;
                let min = pos - Vec2::one() * radius;
                let max = pos + Vec2::one() * radius;
                (Rect::new(min.x, max.x, min.y, max.y), index)
            })
            .collect::<Vec<_>>();

        let mut broccoli = broccoli::Tree::par_new(&mut rects);

        let ptr = self as *mut Self as usize;
        
        broccoli.par_find_colliding_pairs(|i, j| {
            let sim = unsafe { &mut *(ptr as *mut Self) };

            let i = *i.unpack_inner();
            let j = *j.unpack_inner();

            sim.resolve(i, j);
        });
    }

    fn resolve(&mut self, i: usize, j: usize) {
        let b1 = &self.bodies[i];
        let b2 = &self.bodies[j];

        let p1 = b1.pos;
        let p2 = b2.pos;

        let r1 = b1.radius;
        let r2 = b2.radius;

        let d = p2 - p1;
        let r = r1 + r2;

        if d.mag_sq() > r * r {
            return;
        }

        let v1 = b1.vel;
        let v2 = b2.vel;

        let v = v2 - v1;

        let d_dot_v = d.dot(v);

        let m1 = b1.mass;
        let m2 = b2.mass;

        let weight1 = m2 / (m1 + m2);
        let weight2 = m1 / (m1 + m2);

        if d_dot_v >= 0.0 && d != Vec2::zero() {
            let tmp = d * (r / d.mag() - 1.0);
            self.bodies[i].pos -= weight1 * tmp;
            self.bodies[j].pos += weight2 * tmp;
            return;
        }

        let v_sq = v.mag_sq();
        let d_sq = d.mag_sq();
        let r_sq = r * r;

        let t = (d_dot_v + (d_dot_v * d_dot_v - v_sq * (d_sq - r_sq)).max(0.0).sqrt()) / v_sq;

        self.bodies[i].pos -= v1 * t;
        self.bodies[j].pos -= v2 * t;

        let p1 = self.bodies[i].pos;
        let p2 = self.bodies[j].pos;
        let d = p2 - p1;
        let d_dot_v = d.dot(v);
        let d_sq = d.mag_sq();

        let tmp = d * (1.5 * d_dot_v / d_sq);
        let v1 = v1 + tmp * weight1;
        let v2 = v2 - tmp * weight2;

        self.bodies[i].vel = v1;
        self.bodies[j].vel = v2;
        self.bodies[i].pos += v1 * t;
        self.bodies[j].pos += v2 * t;
    }

    pub fn update_colors(&mut self) {
        // Find min and max velocity magnitude for normalization
        let (mut min_v, mut max_v) = (f32::MAX, f32::MIN);
        for body in &self.bodies {
            let v = body.vel.mag();
            if v < min_v { min_v = v; }
            if v > max_v { max_v = v; }
        }
        let range = (max_v - min_v).max(1e-5);

        for (i, body) in self.bodies.iter_mut().enumerate() {
            if i == 0 {
                // Keep center body black
                continue;
            }
            let v = body.vel.mag();
            let t = ((v - min_v) / range).clamp(0.0, 1.0);
            // Map t in [0,1] to color (blue -> cyan -> green -> yellow -> red)
            // Hue from 240 (blue) to 0 (red)
            let hue = 240.0 * (1.0 - t);
            let (r, g, b) = Self::hsv_to_rgb(hue, 1.0, 1.0);
            body.color = [r, g, b];
        }
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
        let c = v * s;
        let h_ = h / 60.0;
        let x = c * (1.0 - ((h_ % 2.0) - 1.0).abs());
        let (r1, g1, b1) = match h_ as u32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 | _ => (c, 0.0, x),
        };
        let m = v - c;
        (r1 + m, g1 + m, b1 + m)
    }
}
