domain_size = [6, 4.5]
particle_spacing = 0.2
particle_mass = 1.0
viscosity = 0.001
gravity = [0, -9.81]
time_step = 0.01
steps = 200
u_scale = 0.2

class Particle:
    def __init__(self, position, velocity, mass, density=1.0, is_boundary=False):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.mass = mass
        self.density = density
        self.force = np.zeros(2, dtype=np.float64)
        self.is_boundary = is_boundary

class SPHSimulation:
    def __init__(self, context, domain_size, particle_spacing, particle_mass, viscosity, gravity, time_step):
        self.context = context
        self.domain_size = domain_size
        self.particle_spacing = particle_spacing
        self.particle_mass = particle_mass
        self.viscosity = viscosity
        self.gravity = np.array(gravity)
        self.time_step = time_step
        self.particles = self.initialize_particles()

    def initialize_particles(self):
        Particle = self.context.get("Particle")
        particles = []
        # internal domain
        for x in np.arange(0, 2, self.particle_spacing):
            for y in np.arange(0, 4, self.particle_spacing):
                position = [x, y]
                #velocity = [5e-1*((y/4)**2+(x-2)**2)/2, 0]
                velocity = [0, 0]
                particles.append(Particle(position, velocity, self.particle_mass))
        # boundary
        for x in np.arange(-2 * self.particle_spacing, self.domain_size[0] + 2 * self.particle_spacing, self.particle_spacing):
            for y in np.arange(-2 * self.particle_spacing, self.domain_size[1] + 2 * self.particle_spacing, self.particle_spacing):
                if x < 0 or x > self.domain_size[0] or y < 0 or y > self.domain_size[1]:
                    position = [x, y]
                    velocity = [0, 0]
                    particles.append(Particle(position, velocity, self.particle_mass, is_boundary=True))
        for x in np.arange(self.domain_size[0]/1.8,self.domain_size[0]/1.8+2*self.particle_spacing, self.particle_spacing ):
            for y in np.arange(0, 4*self.particle_spacing, self.particle_spacing):
                position = [x, y]
                velocity = [0, 0]
                particles.append(Particle(position, velocity, self.particle_mass, is_boundary=True))
        return particles

    def compute_density(self):
        for p in self.particles:
            p.density = sum(self.kernel(p.position - q.position) for q in self.particles)

    def compute_forces(self):
        for p in self.particles:
            p.force = self.gravity * p.mass
            for q in self.particles:
                if p != q:
                    r = p.position - q.position
                    p.force += self.viscosity * (q.velocity - p.velocity) * self.kernel_laplacian(r)

    def compute_pressure_forces(self):
        for p in self.particles:
            pressure_force = np.zeros(2)
            for q in self.particles:
                if p != q:
                    r = p.position - q.position
                    pressure_force += (p.density + q.density) / 2 * self.kernel_gradient(r)
            p.force -= pressure_force

    def update_velocities(self):
        for p in self.particles:
            if not p.is_boundary:
                p.velocity += self.time_step * p.force / p.density
                self.apply_boundary_conditions(p)

    def apply_boundary_conditions(self, p):
        for q in self.particles:
            if q.is_boundary:
                r = p.position - q.position
                distance = np.linalg.norm(r)
                if distance < self.particle_spacing and distance != 0:
                    normal = r / distance
                    if np.dot(p.velocity, normal) < 0:
                        p.velocity -= np.dot(p.velocity, normal) * normal

    def update_positions(self):
        for p in self.particles:
            p.position += self.time_step * p.velocity

    def kernel(self, r):
        h = 2*self.particle_spacing
        q = np.linalg.norm(r) / h
        if q <= 1:
            return (1 - q) ** 3
        else:
            return 0

    def kernel_gradient(self, r):
        h = 2*self.particle_spacing
        q = np.linalg.norm(r) / h
        if q <= 1:
            return -3 * (1 - q) ** 2 * r / (h * np.linalg.norm(r))
        else:
            return np.zeros(2)

    def kernel_laplacian(self, r):
        h = 2*self.particle_spacing
        q = np.linalg.norm(r) / h
        if q <= 1:
            return 6 * (1 - q) / (h ** 2)
        else:
            return 0

    def step(self):
        self.compute_density()
        self.compute_forces()
        self.compute_pressure_forces()
        self.update_velocities()
        self.update_positions()

def sph_step(self, cfg, context):
    rt1 = Text(f"f(x)", color=self.text_color, font_size=self.m_size).shift(2*LEFT)
    bg1 = SurroundingRectangle(rt1, color=self.main_color, fill_opacity=self.opacity, buff=self.box_buff)
    pnts_grp = Group()
    r_grp = Group()
    for _ in range(10):
        xx = (random.random() * 1.5) - 0.75
        yy = (random.random() * 1.5) - 0.52
        pnts_grp.add(Dot(color=self.main_color).shift(xx*RIGHT+yy*UP))
        cr = Circle(radius=0.7, color=self.secondary_color, fill_opacity=self.opacity).shift(xx*RIGHT+yy*UP)
        cr.set_stroke(width=0)
        r_grp.add(cr)
    pnts_grp = pnts_grp.shift(2*RIGHT)
    r_grp = r_grp.shift(2*RIGHT)
    a = Arrow(bg1.get_right(), r_grp.get_left()*np.array([1, 0.0, 0.0]), buff=0.1, color=self.main_color)
    all_grp = Group(rt1, bg1, r_grp, pnts_grp, a)
    self.play(
        FadeIn(rt1, bg1, run_time=self.fadein_rt),
        FadeIn(r_grp, pnts_grp, run_time=self.fadein_rt),
        FadeIn(a, run_time=self.fadein_rt))
    self.next_slide()
    self.play(all_grp.animate().shift(2*UP))
    circle_txt = Tex('$$\quad f_i \quad m_i \quad$$', font_size=self.b_size)
    circle = Circle(color=self.main_color, fill_opacity=self.opacity/2.0).surround(circle_txt)
    circle_grp = VGroup(circle, circle_txt).shift(LEFT)
    cr = Circle(radius=0.7, color=self.secondary_color, fill_opacity=self.opacity)
    cr.set_stroke(width=0)
    kernel_grp = VGroup(Tex('$$\mathbf{W}$$', font_size=self.b_size), cr)
    kernel_grp = kernel_grp.shift(RIGHT)
    old_circle = all_grp[3][0].copy()
    old_kernel = all_grp[2][0].copy()
    self.play(
        Transform(old_circle, circle_grp, run_time=self.transform_rt),
        Transform(old_kernel, kernel_grp, run_time=self.transform_rt),
    )
    self.next_slide()
    txt = Text(f"Kernels").shift(3*RIGHT)
    self.last = txt
    self.play(FadeIn(txt, run_time=self.fadein_rt))
    self.items_step(cfg.kernel_items, self.last)
    txt = Text(f"Approximations").shift(3*LEFT)
    f_app = Tex('$$F(\mathbf{x}_i) = \sum_{j}{F_j \\frac{m_j}{\\rho_j} W_{ij}}$$').next_to(txt, 1.2*DOWN)
    rho_app = Tex('$$\\rho(\mathbf{x}_i) = \sum_{j}{m_j W_{ij}}$$').next_to(f_app, 0.7*DOWN)
    der = Text("Derivatives shift to kernels").next_to(rho_app, 0.7*DOWN)
    self.play(FadeIn(txt, f_app, rho_app, der, run_time=self.fadein_rt))
    self.reset_step(None, None)
    Particle = context.get('Particle')
    SPHSimulation = context.get('SPHSimulation')
    simulation = SPHSimulation(
        context,
        context.get("domain_size"),
        context.get("particle_spacing"),
        context.get("particle_mass"),
        context.get("viscosity"),
        context.get("gravity"),
        context.get("time_step"),
    )
    particles = VGroup()
    velocities = VGroup()
    graph_shift = 2.5*DOWN
    time_shift = 4.8*RIGHT+UP

    domain_size = context.get("domain_size")
    spacing = context.get("particle_spacing")/2
    time = Text(f"Time: {0:05.2f}").shift(time_shift)
    b1s = domain_size[1]*UP+spacing*(UP+LEFT)
    b1e = spacing*(DOWN+LEFT)
    b2e = domain_size[0]*RIGHT+spacing*(DOWN+RIGHT)
    b3e = domain_size[0]*RIGHT+domain_size[1]*UP+spacing*(UP+RIGHT)
    b1 = Line(b1s, b1e, stroke_width=5, color=self.warn_color)
    b2 = Line(b1e, b2e, stroke_width=5, color=self.warn_color)
    b3 = Line(b2e, b3e, stroke_width=5, color=self.warn_color)
    b4 = Rectangle(width=4*spacing, height=8*spacing, color=self.warn_color)
    b44 = BackgroundRectangle(b4, color=self.warn_color, fill_opacity=1).next_to(b2, 0).shift(4*spacing*RIGHT+4*spacing*UP)
    n1 = Arrow((b1s+b1e)/2, (b1s+b1e)/2+0.5*RIGHT, buff=0.0, color=self.warn_color)
    n2 = Arrow((b1e+b2e)/3, (b1e+b2e)/3+0.5*UP, buff=0.0, color=self.warn_color)
    n3 = Arrow((b2e+b3e)/2, (b2e+b3e)/2+0.5*LEFT, buff=0.0, color=self.warn_color)
    omega = Tex("$$\Omega: \mathbf{v}\cdot\mathbf{n}_\Omega \geq 0$$", color=self.warn_color).shift(b2.get_center()+0.5*DOWN)
    nn = Tex("$\mathbf{n}$", color=self.warn_color).next_to(n2, 0.2*UP)
    bounds = Group(b1,b2,b3,b44, n1,n2,n3,nn, omega)
    self.play(
        FadeIn(bounds.shift(graph_shift), run_time=self.fadein_rt),
        FadeIn(time, run_time=self.fadein_rt),
    )
    self.next_slide()

    mom_eqn = "$$\\rho\\frac{D\mathbf{v}}{Dt} = - \mathbf{\\nabla}(p) + \mu \\nabla^2\mathbf{v} + \\rho\mathbf{g}, \quad \\frac{D\\rho}{Dt} = 0$$"
    eqns_1 = Tex(mom_eqn, font_size=self.b_size).next_to(self.layout[0], 3*DOWN).align_to(self.layout[0], LEFT)
    pos_eqn = "$$\\frac{D\mathbf{x}}{Dt} = \mathbf{v}$$"
    eqns_2 = Tex(pos_eqn, font_size=self.b_size).next_to(eqns_1, DOWN).align_to(eqns_1, LEFT)

    cond = r"\quad q = \frac{\|\mathbf{r}\|}{h}"
    kernel = r"W(\mathbf{r}) = \begin{cases} (1 - q)^3 & \text{if } q \leq 1 \\ 0 & \text{otherwise} \end{cases}"
    grad_kernel = r"\mathbf{\nabla} W(\mathbf{r}) = \begin{cases} -3 (1 - q)^2 \frac{\mathbf{r}}{h \|\mathbf{r}\|} & \text{if } q \leq 1 \\ \vec{0} & \text{otherwise} \end{cases}"
    lapl_kernel = r"\nabla^2 W(\mathbf{r}) = \begin{cases} \frac{6(1 - q)}{h^2} & \text{if } q \leq 1 \\ 0 & \text{otherwise} \end{cases}"
    eqns_3 = MathTex(kernel, cond, font_size=self.b_size).next_to(eqns_2, 2*DOWN).align_to(eqns_2, LEFT)
    eqns_4 = MathTex(grad_kernel, cond, font_size=self.b_size).next_to(eqns_3, DOWN).align_to(eqns_3, LEFT)
    eqns_5 = MathTex(lapl_kernel, cond, font_size=self.b_size).next_to(eqns_4, DOWN).align_to(eqns_4, LEFT)

    self.play(
        FadeIn(eqns_1, eqns_2, eqns_3, eqns_4, eqns_5, run_time=self.fadein_rt),
    )
    self.next_slide()
    self.play(
        FadeOut(eqns_1, eqns_2, eqns_3, eqns_4, eqns_5, bounds, run_time=self.fadeout_rt),
    )
    for p in simulation.particles:
        color = self.important_color if p.is_boundary else self.main_color
        start = p.position[0]*RIGHT+p.position[1]*UP
        particles.add(Dot(color=color).shift(start))
        end = start + context.get("u_scale")*p.velocity[0]*RIGHT + context.get("u_scale")*p.velocity[1]*UP
        arr = Arrow(start, end, buff=0.0, color=self.secondary_color)
        arr.set_opacity(0.3)
        velocities.add(arr)
    self.play(
        FadeIn(velocities.shift(graph_shift), run_time=self.fadein_rt),
        FadeIn(particles.shift(graph_shift), run_time=self.fadein_rt),
    )

    algo_0 = Text(f"A SIMPLISTIC approach: ").next_to(self.layout[0], 2.5*DOWN).align_to(self.layout[0], LEFT)
    algo_1 = Text(f"- Reconstruct densities: ").next_to(algo_0, DOWN).align_to(algo_0, LEFT)
    algo_2 = MathTex(r"\rho_i = \Sigma_j m_jW_{ij}", font_size=self.m_size).next_to(algo_1, 0.5*RIGHT)
    algo_3 = Text(f"- Compute viscous forces: ").next_to(algo_1, DOWN).align_to(algo_1, LEFT)
    algo_4 = MathTex(r"m_i\nu\Sigma_j(\mathbf{v_j}-\mathbf{v_i})\nabla^2 W_{ij}", font_size=self.m_size).next_to(algo_3, 0.5*RIGHT)
    algo_5 = Text(f"- Compute external forces: ").next_to(algo_3, DOWN).align_to(algo_3, LEFT)
    algo_6 = MathTex(r"m_i \mathbf{g}", font_size=self.m_size).next_to(algo_5, 0.5*RIGHT)
    algo_7 = Text(f"- Update velocities: ").next_to(algo_5, DOWN).align_to(algo_5, LEFT)
    algo_8 = MathTex(r"\mathbf{v}_i^{*} = \mathbf{v}_i + \frac{\Delta t}{m_i}(\mathbf{F}_i^\nu + \mathbf{F}_i^g)", font_size=self.m_size).next_to(algo_7, 0.5*RIGHT)
    algo_9 = Text(f"- Compute pressure forces: ").next_to(algo_7, DOWN).align_to(algo_7, LEFT)
    algo_10 = MathTex(r"\Sigma_j \frac{\rho_i+\rho_j}{2}\mathbf{\nabla W_{ij}}", font_size=self.m_size).next_to(algo_9, 0.5*RIGHT)
    algo_11 = Text(f"- Correct velocities and positions:").next_to(algo_9, DOWN).align_to(algo_9, LEFT)
    algo_12 = MathTex(r"\mathbf{v}_{i, t+\Delta t} = \mathbf{v}_i^* + \frac{\Delta t}{m_i}\mathbf{F}_i^p", font_size=self.m_size).next_to(algo_11, DOWN)
    algo_13 = MathTex(r"\mathbf{x}_{i, t+\Delta t} = \mathbf{x}_i + \Delta t\mathbf{v}_{i,t+\Delta t}", font_size=self.m_size).next_to(algo_12, DOWN)

    self.play(
        FadeIn(
            algo_0,
            algo_1, algo_2,
            algo_3, algo_4,
            algo_5, algo_6,
            algo_7, algo_8,
            algo_9, algo_10,
            algo_11, algo_12, algo_13,
            run_time=self.fadein_rt)
    )

    self.next_slide()
    steps = context.get("steps")
    u_scale = context.get("u_scale")
    for step in range(steps):
        simulation.step()
        new_particles = VGroup()
        new_velocities = VGroup()
        for i in range(len(simulation.particles)):
            pos = simulation.particles[i].position
            new_pos = pos[0]*RIGHT+pos[1]*UP
            color = self.important_color if simulation.particles[i].is_boundary else self.main_color
            new_particles.add(Dot(color=color).shift(new_pos))
            new_end = new_pos + u_scale*simulation.particles[i].velocity[0]*RIGHT + u_scale*simulation.particles[i].velocity[1]*UP
            arr = Arrow(new_pos, new_end, buff=0.0, color=self.secondary_color)
            arr.set_opacity(0.3)
            new_velocities.add(arr)
        self.play(
            Transform(velocities, new_velocities.shift(graph_shift), run_time=0.1),
            Transform(particles, new_particles.shift(graph_shift), run_time=0.1),
            Transform(time, Text(f"Time: {(1+step)*context.get('time_step'):05.2f}").shift(time_shift), run_time=0.1),
        )

    self.next_slide()
    self.keep_only_objects(self.layout)

    cons_0 = Text(f"Nice visuals and all, but...").next_to(self.layout[0], 2.5*DOWN).align_to(self.layout[0], LEFT)
    #cons_1_math = r"\Sigma_j{F_j\frac{m_j}{\rho_j}W(\mathbf{x_i}-\mathbf{x_j}, h)} \approx \int \frac{F(\mathbf{x}^{'})}{\rho(\mathbf{x}^{'})}W(\mathbf{x}-\mathbf{x^{'}}, h)dm^{'}"
    #cons_1 = MathTex(cons_1_math, font_size=self.b_size).next_to(cons_0, DOWN).align_to(cons_0, LEFT)
    order_0 = r"\Sigma_j{\frac{m_j}{\rho_j}W_{ij}}"
    order_1 = r"\Sigma_j{\frac{m_j}{\rho_j}(\mathbf{x}_j-\mathbf{x}_i)W_{ij}}"
    cons_2_math = r"\Sigma_j{F_j\frac{m_j}{\rho_j}W(\mathbf{x_i}-\mathbf{x_j}, h)} = F_i " + order_0 + r" + \nabla F_i \cdot " + order_1 + r" + \mathcal{O}(||\mathbf{r}||^2)"
    cons_2 = MathTex(cons_2_math, font_size=self.b_size, tex_to_color_map={order_0: self.main_color, order_1: self.warn_color}).next_to(cons_0, DOWN).align_to(cons_0, LEFT)
    self.play(
        FadeIn(cons_0, cons_2,
               run_time=self.fadein_rt)
    )
    self.last = cons_2
    self.items_step(cfg.consistency_items, self.last)
