//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Benyovszki Patrik
// Neptun : IQ00CM
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"



// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix

	layout(location = 0) in vec4 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec3 col;

	out vec3 color;

	void main() {
		color = col;
		gl_Position = vec4(vp.x*vp.z, vp.y*vp.z, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	in vec3 color;			// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";



GPUProgram gpuProgram; // vertex and fragment shaders
const int nTesselatedVertices = 40;
vec2 origo = vec2(0, 0);
float elapsedTimeSinceLastOnIdle = 0;


/*
https://stackoverflow.com/questions/686353/random-float-number-generation?rq=1
*/
float randomNum(float LO, float HI) {
	return  LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

float translateToHyperbolic(vec2 coords) {
	float w = sqrt(1 + coords.x * coords.x + coords.y * coords.y);
	return 1 / (w + 1);
}

void translateOrigo(vec2 t) {
	origo = origo + t;
}

struct Atom {
	int relative_mass;
	int relative_charge;
	vec2 pos;

public:
	Atom(int rel_charge = 0) {
		if (rel_charge != 0)
			relative_charge = rel_charge;
		else {
			if (rand() % 2 == 0)
				relative_charge = rand() % 256;
			else
				relative_charge = -(rand() % 256);
		}
		relative_mass = rand() % 10 + 1;

		pos.x = randomNum(-5, 5);
		pos.y = randomNum(-5, 5);
	}

	void draw(unsigned int vbo[2], mat4 M, vec2 center) {
		
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		vec4 vertices[nTesselatedVertices];
		for (int i = 0; i < nTesselatedVertices; ++i) {
			float fi = i * 2 * M_PI / nTesselatedVertices;
			vertices[i] = vec4(cosf(fi) * relative_mass / 30 + pos.x, sinf(fi) * relative_mass / 30 + pos.y, 0, 1) * M;
			vertices[i].z = translateToHyperbolic(vec2(vertices[i].x, vertices[i].y));
		}
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec4) * nTesselatedVertices, vertices, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		float vertexColor[3 * nTesselatedVertices];
		for (int i = 0; i < 3 * nTesselatedVertices; ++i)
			vertexColor[i] = 0;

		for (int i = 0; i < nTesselatedVertices; ++i) {
			if (relative_charge < 0)
				vertexColor[3 * i + 2] -= (float)relative_charge / 255;
			else
				vertexColor[3 * i] += (float)relative_charge / 255;
		}
		

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColor), vertexColor, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		gpuProgram.setUniform(TranslateMatrix(vec3(0, 0, 0)), "MVP");

		glDrawArrays(GL_TRIANGLE_FAN, 0, nTesselatedVertices);
	}
};

struct Bond {
	Atom a1, a2;

public:
	
	Bond(Atom at1, Atom at2) {
		a1 = at1;
		a2 = at2;
	}

	void draw(unsigned int vbo[2], mat4 M, vec2 center) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		vec4 vertices[nTesselatedVertices];
		vec2 diff = (a2.pos - a1.pos) / (nTesselatedVertices - 1);
		vec2 actual = a1.pos;
		for (int i = 0; i < nTesselatedVertices; ++i) {
			vertices[i] = vec4(actual.x, actual.y, 1, 1) * M;
			vertices[i].z = translateToHyperbolic(vec2(vertices[i].x, vertices[i].y));
			actual = actual + diff;
		}
		
		glBufferData(GL_ARRAY_BUFFER, nTesselatedVertices * sizeof(vec4), vertices, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		float vertexColor[nTesselatedVertices * 3];
		for (int i = 0; i < 3 * nTesselatedVertices; ++i)
			vertexColor[i] = 1;


		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColor), vertexColor, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		gpuProgram.setUniform(TranslateMatrix(vec3(0, 0, 0)), "MVP");

		glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
	}
};

struct Molecule {
	std::vector<Atom> atoms;
	std::vector<Bond> bonds;
	vec2 center;
	float theta;
	int relative_massSum;
	unsigned int vao;
	unsigned int vbo[2];
	float phi;
	vec3 v, w;

public:
	
	void create() {
		phi = 0;
		v = vec3(0, 0, 0);
		w = vec3(0, 0, 0);
		bonds.clear();
		atoms.clear();
		int atomNum = rand() % 7 + 2;
		int chargeSum;

		atoms.push_back(Atom());
		
		chargeSum = atoms.back().relative_charge;

		center = atoms.back().pos;
		relative_massSum = atoms.back().relative_mass;
		
		for (int i = 2; i < atomNum; ++i) {
			atoms.push_back(Atom());
			
			chargeSum += atoms.back().relative_charge;

			calcCenter();
			relative_massSum += atoms.back().relative_mass;

			int pairNum = rand() % (atoms.size() - 1);
			bonds.push_back(Bond(atoms.back(), atoms.at(pairNum)));
		}
		atoms.push_back(Atom(-chargeSum));
		
		calcCenter();

		int pairNum = rand() % (atoms.size() - 1);
		bonds.push_back(Bond(atoms.back(), atoms.at(pairNum)));

		translateToOrigo();

		calcTheta();

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2, &vbo[0]);

	}

	void calcCenter() {
		center = (relative_massSum * center + atoms.back().relative_mass * atoms.back().pos) / (relative_massSum + atoms.back().relative_mass);
	}

	void translateToOrigo() {
		for (int i = 0; i < atoms.size(); ++i)
			atoms.at(i).pos = atoms.at(i).pos - center;

		for (int i = 0; i < bonds.size(); ++i) {
			bonds.at(i).a1.pos = bonds.at(i).a1.pos - center;
			bonds.at(i).a2.pos = bonds.at(i).a2.pos - center;
		}
	}

	void calcTheta() {
		theta = 0;
		for (int i = 0; i < atoms.size(); ++i)
			theta += atoms.at(i).relative_mass * (atoms.at(i).pos.x * atoms.at(i).pos.x + atoms.at(i).pos.y * atoms.at(i).pos.y);
	}

	void draw() {
		glBindVertexArray(vao);
 		for (int i = 0; i < bonds.size(); ++i) {
			bonds.at(i).draw(vbo, M(), center);
		}

		for (int i = 0; i < atoms.size(); ++i) {
			atoms.at(i).draw(vbo, M(), center);
		}
	}

	mat4 M() {
		return TranslateMatrix(center - origo);
	}

	void updatePoses() {
		for (int i = 0; i < atoms.size(); ++i) {
			vec4 tmp = vec4(atoms[i].pos.x, atoms[i].pos.y, 0, 0) * 
				mat4(cosf(phi), sinf(phi), 0, 0,
					-sinf(phi), cosf(phi), 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 1
			);
			atoms[i].pos = vec2(tmp.x, tmp.y);
		}
		for (int i = 0; i < bonds.size(); ++i) {
			vec4 tmp = vec4(bonds[i].a1.pos.x, bonds[i].a1.pos.y, 0, 0) *
				mat4(cosf(phi), sinf(phi), 0, 0,
					-sinf(phi), cosf(phi), 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 1
				);
			bonds[i].a1.pos = vec2(tmp.x, tmp.y);

			tmp = vec4(bonds[i].a2.pos.x, bonds[i].a2.pos.y, 0, 0) *
				mat4(cosf(phi), sinf(phi), 0, 0,
					-sinf(phi), cosf(phi), 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 1
				);
			bonds[i].a2.pos = vec2(tmp.x, tmp.y);
		}
	}
};

Molecule m1, m2;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.5f, 0.5f);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
	m1.draw();
	m2.draw();
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {

	if (key == ' ') {
		m1.create();
		
		m2.create();
		

		origo = vec2(0, 0);
	}

	if (key == 'e') {
		translateOrigo(vec2(0, -0.1f));
	}

	if (key == 'd') {
		translateOrigo(vec2(-0.1f, 0));
	}

	if (key == 's') {
		translateOrigo(vec2(0.1f, 0));
	}

	if (key == 'x') {
		translateOrigo(vec2(0, 0.1f));
	}

	glutPostRedisplay();
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;
	
	vec3 sumF1;
	vec3 sumF2;
	vec3 Fij;
	vec3 sumM1;
	vec3 sumM2;
	
	for (float t = elapsedTimeSinceLastOnIdle; t < sec; t += 0.01) {
		sumF1 = vec3(0, 0, 0);
		sumF2 = vec3(0, 0, 0);
		sumM1 = vec3(0, 0, 0);
		sumM2 = vec3(0, 0, 0);
		float dt = 0.01;
		for(int i = 0; i < m1.atoms.size(); ++i)
			for (int j = 0; j < m2.atoms.size(); ++j) {
				Atom tmp1 = m1.atoms[i];
				Atom tmp2 = m2.atoms[j];
				Fij = (tmp1.relative_charge * tmp2.relative_charge) / (200 * M_PI * length(tmp1.pos + m1.center - tmp2.pos + m2.center)) * 
					normalize(vec3(tmp1.pos.x + m1.center.x, tmp1.pos.y + m1.center.y, 0) - vec3(tmp2.pos.x + m2.center.x, tmp2.pos.y + m2.center.y, 0));
				sumF1 = sumF1 + Fij;
				sumM1 = sumM1 + cross(vec3(tmp1.pos.x, tmp1.pos.y, 0), Fij);
				sumM2 = sumM2 + cross(vec3(tmp2.pos.x, tmp2.pos.y, 0), -Fij);
			}
		sumF2 = -sumF1;
		
		sumF1 = sumF1 - 100 * m1.v;
		sumF2 = sumF2 - 100 * m2.v;	
		sumM1 = sumM1 - 100 * m1.w;
		sumM2 = sumM2 - 100 * m2.w;


		m1.v = m1.v + sumF1 / m1.relative_massSum * dt;
		m2.v = m2.v + sumF2 / m2.relative_massSum * dt;

		m1.center = m1.center + vec2((m1.v * dt).x, (m1.v * dt).y);
		m2.center = m2.center + vec2((m2.v * dt).x, (m2.v * dt).y);

		m1.w = m1.w + sumM1 / m1.theta * dt;
		m2.w = m2.w + sumM2 / m2.theta * dt;

		m1.phi = (m1.w * dt).z;
		m2.phi = (m2.w * dt).z;

		m1.updatePoses();
		m2.updatePoses();
		glutPostRedisplay();
	}

	elapsedTimeSinceLastOnIdle = sec;
}

