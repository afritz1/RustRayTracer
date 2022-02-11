use image;
use num_cpus;
use rand::Rng;
use std::fmt;
use std::path::Path;
use std::thread;

/*
Feb. 2022
rust rt goals:
- learn rust basics
- get something on-screen that's ray traced
- new RNG seed every start
*/

// @todo: fix super sampling, seems too dark on some edges
// @todo: fix multi-threading, don't understand Rust's scope management yet (shade_frame_buffer())
// @todo: add more shading, lights, shadows, and reflections

const SCREEN_WIDTH : u32 = 1280;
const SCREEN_HEIGHT : u32 = 720;
const SUPER_SAMPLES : u32 = 2;
const SPHERE_COUNT : u32 = 100;
const MAX_SAMPLE_COUNT : u32 = 512;
const FOV_Y : f64 = 55.0;

const PIXEL_COUNT : usize = (SCREEN_WIDTH * SCREEN_HEIGHT) as usize;

const SCREEN_WIDTH_REAL : f64 = SCREEN_WIDTH as f64;
const SCREEN_HEIGHT_REAL : f64 = SCREEN_HEIGHT as f64;
const ASPECT_RATIO : f64 = SCREEN_WIDTH_REAL / SCREEN_HEIGHT_REAL;

const SUPER_SAMPLES_REAL : f64 = SUPER_SAMPLES as f64;
const SUPER_SAMPLES_REAL_SQR : f64 = SUPER_SAMPLES_REAL * SUPER_SAMPLES_REAL;

const PI : f64 = 3.1415926535897932;
const EPSILON : f64 = 1.0e-3;
const DEG_TO_RAD : f64 = PI / 180.0;

fn rand_percent() -> f64
{
    rand::thread_rng().gen::<f64>()
}

#[derive(Copy, Clone)]
struct Vec3
{
    x : f64,
    y : f64,
    z : f64
}

impl Vec3
{
    const ZERO : Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    const UNIT_X : Vec3 = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    const UNIT_Y : Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    const UNIT_Z : Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

    fn new(x: f64, y: f64, z: f64) -> Vec3
    {
        Vec3 { x: x, y: y, z: z }
    }

    fn add(&self, other: &Vec3) -> Vec3
    {
        Vec3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    fn subtract(&self, other: &Vec3) -> Vec3
    {
        Vec3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    fn negate(&self) -> Vec3
    {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }

    fn multiply_scalar(&self, s: f64) -> Vec3
    {
        Vec3 { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    fn multiply_vector(&self, v: &Vec3) -> Vec3
    {
        Vec3 { x: self.x * v.x, y: self.y * v.y, z: self.z * v.z }
    }

    fn divide(&self, s: f64) -> Vec3
    {
        let recip = 1.0 / s;
        Vec3 { x: self.x * recip, y: self.y * recip, z: self.z * recip }
    }

    fn length_sqr(&self) -> f64
    {
        (self.x * self.x) + (self.y * self.y) + (self.z * self.z)
    }

    fn length(&self) -> f64
    {
        f64::sqrt(self.length_sqr())
    }

    fn normalized(&self) -> Vec3
    {
        let len = self.length();
        let len_recip = 1.0 / len;
        Vec3::new(self.x * len_recip, self.y * len_recip, self.z * len_recip)
    }

    fn dot(&self, other: &Vec3) -> f64
    {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    }

    fn cross(&self, other: &Vec3) -> Vec3
    {
        Vec3::new(
            (self.y * other.z) - (other.y * self.z),
		    (other.x * self.z) - (self.x * other.z),
		    (self.x * other.y) - (other.x * self.y))
    }

    fn lerp(&self, other: &Vec3, percent: f64) -> Vec3
    {
        Vec3::new(
            self.x + ((other.x - self.x) * percent),
            self.y + ((other.y - self.y) * percent),
            self.z + ((other.z - self.z) * percent))
    }

    fn clamped01(&self) -> Vec3
    {
        Vec3::new(
            num::clamp(self.x, 0.0, 1.0),
            num::clamp(self.y, 0.0, 1.0),
            num::clamp(self.z, 0.0, 1.0))
    }

    fn random_color() -> Vec3
    {
        let x = rand_percent();
        let y = rand_percent();
        let z = rand_percent();
        Vec3::new(x, y, z)
    }

    fn random_dir() -> Vec3
    {
        fn rand_func() -> f64 { (2.0 * rand_percent()) - 1.0 }
        let x = rand_func();
        let y = rand_func();
        let z = rand_func();
        Vec3::new(x, y, z).normalized()
    }

    fn random_hemisphere_dir(normal: &Vec3) -> Vec3
    {
        let mut dir = Vec3::random_dir();
        if normal.dot(&dir) >= 0.0
        {
            dir
        }
        else
        {
            dir.negate()
        }
    }
}

impl fmt::Display for Vec3
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        write!(f, "{}, {}, {}", self.x, self.y, self.z)
    }
}

#[derive(Copy, Clone)]
struct Camera
{
    eye : Vec3,
    dir : Vec3,
    right : Vec3,
    up : Vec3
}

impl Camera
{
    fn new(eye: &Vec3, target: &Vec3) -> Camera
    {
        let dir = target.subtract(eye).normalized();
        let up = Vec3::UNIT_Y;
        let right = dir.cross(&up).normalized();

        Camera
        {
            eye: *eye, dir: dir, right: right.multiply_scalar(ASPECT_RATIO), up: up
        }
    }

    fn make_image_dir(&self, x_percent: f64, y_percent: f64) -> Vec3
    {
        let zoom : f64 = 1.0 / f64::tan((FOV_Y * 0.50) * DEG_TO_RAD);
        let scaled_fwd = self.dir.multiply_scalar(zoom);
        let scaled_right = self.right.multiply_scalar((2.0 * x_percent) - 1.0);
        let scaled_up = self.up.multiply_scalar((2.0 * y_percent) - 1.0);
        scaled_fwd.add(&scaled_right.subtract(&scaled_up)).normalized()
    }
}

#[derive(Copy, Clone)]
struct Ray
{
    point: Vec3,
    dir: Vec3
}

impl Ray
{
    fn new(point: &Vec3, dir: &Vec3) -> Ray
    {
        Ray { point: *point, dir: *dir }
    }

    fn at(&self, t: f64) -> Vec3
    {
        self.point.add(&self.dir.multiply_scalar(t))
    }
}

#[derive(Copy, Clone)]
struct Sphere
{
    point: Vec3,
    radius: f64,
    radius_sqr: f64,
    radius_recip: f64,
    color: Vec3
}

impl Sphere
{
    fn new(point: &Vec3, radius: f64) -> Sphere
    {
        Sphere
        {
            point: *point,
            radius: radius,
            radius_sqr: radius * radius,
            radius_recip: 1.0 / radius,
            color: Vec3::random_color()
        }
    }

    fn try_hit(&self, ray: &Ray, out_t: &mut f64, out_point: &mut Vec3, out_normal: &mut Vec3) -> bool
    {
        let point_diff = self.point.subtract(&ray.point);
        let ray_dot = point_diff.dot(&ray.dir);
        let determinant = (ray_dot * ray_dot) - point_diff.dot(&point_diff) + self.radius_sqr;
        if determinant < 0.0
        {
            return false;
        }
        else
        {
            let determinant_sqrt = f64::sqrt(determinant);
            let b_minus = ray_dot - determinant_sqrt;
            let b_plus = ray_dot + determinant_sqrt;
            
            if b_minus > EPSILON
            {
                *out_t = b_minus;
            }
            else if b_plus > EPSILON
            {
                *out_t = b_plus;
            }
            else
            {
                return false;
            }

            *out_point = ray.at(*out_t);
            *out_normal = out_point.subtract(&self.point).multiply_scalar(self.radius_recip);
            return true;
        }
    }
}

fn get_ambient_percent(point: &Vec3, normal: &Vec3, spheres: &Vec<Sphere>) -> f64
{
    let point_norm_epsilon = point.add(&normal.multiply_scalar(EPSILON));    
    let mut vis_sample_count = 0;

    for n in 0..MAX_SAMPLE_COUNT
    {
        let hemisphere_dir = Vec3::random_hemisphere_dir(&normal);
        let hemisphere_ray = Ray::new(&point_norm_epsilon, &hemisphere_dir);

        let mut is_occluded = false;        
        for i in 0..spheres.len()
        {
            let sphere = &spheres[i];
            let mut cur_t = f64::INFINITY;
            let mut cur_hit_point = Vec3::ZERO;
            let mut cur_hit_normal = Vec3::ZERO;
            if sphere.try_hit(&hemisphere_ray, &mut cur_t, &mut cur_hit_point, &mut cur_hit_normal)
            {
               is_occluded = true;
               break;
            }
        }

        if !is_occluded
        {
            vis_sample_count = vis_sample_count + 1;
        }
    }

    num::clamp((vis_sample_count as f64) / (MAX_SAMPLE_COUNT as f64), 0.0, 1.0)
}

fn get_shaded_hit(sphere: &Sphere, t: f64, hit_point: &Vec3, hit_normal: &Vec3, spheres: &Vec<Sphere>) -> Vec3
{
    let ambient_percent = get_ambient_percent(&hit_point, &hit_normal, &spheres);
    sphere.color.multiply_scalar(ambient_percent)
}

fn get_sample_color(camera: &Camera, x_percent: f64, y_percent: f64, spheres: &Vec<Sphere>) -> Vec3
{
    let ray = Ray::new(&camera.eye, &camera.make_image_dir(x_percent, y_percent));
    let bg_color = ray.dir;

    let mut hit_index : Option<usize> = None;
    let mut t = f64::INFINITY;
    let mut hit_point = Vec3::ZERO;
    let mut hit_normal = Vec3::ZERO;

    for i in 0..spheres.len()
    {
        let sphere = &spheres[i];

        let mut cur_t = f64::INFINITY;
        let mut cur_hit_point = Vec3::ZERO;
        let mut cur_hit_normal = Vec3::ZERO;
        if sphere.try_hit(&ray, &mut cur_t, &mut cur_hit_point, &mut cur_hit_normal)
        {
            if hit_index.is_some()
            {
                if cur_t >= t
                {
                    continue;
                }
            }

            hit_index = Some(i);
            t = cur_t;
            hit_point = cur_hit_point;
            hit_normal = cur_hit_normal;
        }
    }

    let out_color = match hit_index
    {
        Some(index) => get_shaded_hit(&spheres[index], t, &hit_point, &hit_normal, &spheres),
        None => bg_color
    };

    out_color
}

fn shade_frame_buffer_thread_func(start_y: u32, y_step: u32, camera: &Camera, spheres: &Vec<Sphere>, colors: &mut Vec<u8>)
{
    for y in (start_y..SCREEN_HEIGHT).step_by(y_step as usize)
    {
        let base_y_percent = (y as f64) / SCREEN_HEIGHT_REAL;
        if start_y == 0
        {
            println!("{} / {} ({:.2}%)", y, SCREEN_HEIGHT, (base_y_percent * 100.0));
        }

        for x in 0..SCREEN_WIDTH
        {
            let base_x_percent = (x as f64) / SCREEN_WIDTH_REAL;

            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            for i in 0..SUPER_SAMPLES
            {
                for j in 0..SUPER_SAMPLES
                {
                    let x_percent = base_x_percent + (((i as f64) / SUPER_SAMPLES_REAL) / SCREEN_WIDTH_REAL);
                    let y_percent = base_y_percent + (((j as f64) / SUPER_SAMPLES_REAL) / SCREEN_HEIGHT_REAL);
                    let sample_rgb = get_sample_color(&camera, x_percent, y_percent, &spheres);
                    r = r + sample_rgb.x;
                    g = g + sample_rgb.y;
                    b = b + sample_rgb.z;
                }
            }

            r = r / SUPER_SAMPLES_REAL_SQR;
            g = g / SUPER_SAMPLES_REAL_SQR;
            b = b / SUPER_SAMPLES_REAL_SQR;
            let r_byte = (r * 255.0) as u8;
            let g_byte = (g * 255.0) as u8;
            let b_byte = (b * 255.0) as u8;
            
            let index = ((x + (y * SCREEN_WIDTH)) * 3) as usize;
            colors[index] = r_byte;
            colors[index + 1] = g_byte;
            colors[index + 2] = b_byte;
        }
    }
}

fn shade_frame_buffer(camera: &Camera, spheres: &Vec<Sphere>, colors: &mut Vec<u8>)
{
    shade_frame_buffer_thread_func(0, 1, &camera, &spheres, colors);

    // @todo: get multi-threading working
    /*let cpu_count = num_cpus::get() as u32;
    let threads : Vec<thread::JoinHandle<()>> = Vec::new();
    for i in 0..cpu_count
    {
        let thread_handle = thread::spawn(|| shade_frame_buffer_thread_func(i as u32, cpu_count, &camera, &spheres, &mut colors));
        threads.push(thread_handle);
    }

    for thread_handle in threads
    {
        thread_handle.join().unwrap();
    }*/
}

fn save_to_storage(colors: &Vec<u8>)
{
    let save_path = Path::new("image.png");
    image::save_buffer(&save_path, &colors, SCREEN_WIDTH, SCREEN_HEIGHT, image::ColorType::Rgb8);
}

fn main()
{
    let camera = Camera::new(&Vec3::new(13.0, 1.0, -12.0), &Vec3::new(0.0, 0.5, 0.0));
    println!("Camera: eye{{ {} }}, dir{{ {} }}", camera.eye, camera.dir);

    let mut spheres : Vec<Sphere> = Vec::new();
    for i in 0..SPHERE_COUNT
    {
        let sphere_point = Vec3::random_dir().multiply_scalar(10.0 * rand_percent());
        let sphere_radius = 0.25 + rand_percent();
        let sphere = Sphere::new(&sphere_point, sphere_radius);
        spheres.push(sphere);
    }

    let big_sphere = Sphere::new(&Vec3::new(0.0, -40.0, 0.0), 37.0);
    spheres.push(big_sphere);
    
    let mut colors : Vec<u8> = vec![0; PIXEL_COUNT * 3];

    println!("Processing {}x{}...", SCREEN_WIDTH, SCREEN_HEIGHT);
    shade_frame_buffer(&camera, &spheres, &mut colors);
    
    println!("Saving to storage...");
    save_to_storage(&colors);

    println!("Done.");
}
