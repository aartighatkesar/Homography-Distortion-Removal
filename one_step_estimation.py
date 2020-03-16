from estimate_homography import *

def get_line_equations(pts_1_on_line, pts_2_on_line):
    """

    :param pts_1_on_line: [[x1, y1], [x2, y2]]
    :param pts_2_on_line:
    :return: np.array[[l1, m1, n1], [l2, m2, n2]]
    """

    # Convert to homogenous coordinates

    if isinstance(pts_1_on_line, list):
        pts_1_on_line = np.array(pts_1_on_line)
    if pts_1_on_line.shape[1] != 3:
        pts_1_on_line = np.hstack((pts_1_on_line, np.ones((pts_1_on_line.shape[0], 1))))


    if isinstance(pts_2_on_line, list):
        pts_2_on_line = np.array(pts_2_on_line)
    if pts_2_on_line.shape[1] != 3:
        pts_2_on_line = np.hstack((pts_2_on_line, np.ones((pts_2_on_line.shape[0], 1))))

    lines = np.cross(pts_1_on_line, pts_2_on_line)

    lines = lines/lines[:, -1:]

    return lines


def build_one_step_eqns(perp_lines_1, perp_lines_2):

    # f = 1

    mat_A = np.zeros((perp_lines_1.shape[0], 5)) # a, b, c, d, e
    mat_b = -1 * perp_lines_1[:, -1] * perp_lines_2[:, -1]  # -1(l3 * m3) as f=1

    mat_A[:, 0] = perp_lines_1[:, 0] * perp_lines_2[:, 0]   # l1 * m1
    mat_A[:, 1] = (perp_lines_1[:, 0] * perp_lines_2[:, 1] + perp_lines_1[:, 1] * perp_lines_2[:, 0])/2  # (l1 * m2 +l2 * m1)/2
    mat_A[:, 2] = perp_lines_1[:, 1] * perp_lines_2[:, 1]  # l2 * m2
    mat_A[:, 3] = (perp_lines_1[:, 0] * perp_lines_2[:, 2] + perp_lines_1[:, 2] * perp_lines_2[:, 0])/2  # (l1 * m3 +l3 * m1)/2
    mat_A[:, 4] = (perp_lines_1[:, 1] * perp_lines_2[:, 2] + perp_lines_1[:, 2] * perp_lines_2[:, 1])/2  # (l2 * m3 +l3 * m2)/2

    return mat_A, mat_b


def get_image_deg_conic(perp_lines_1, perp_lines_2):

    mat_A, mat_b = build_one_step_eqns(perp_lines_1, perp_lines_2)

    C_flat = np.matmul(np.linalg.pinv(mat_A), mat_b) # a:0, b:1, c:2, d:3, e:4

    C_dash_inf = np.array([[C_flat[0], C_flat[1]/2, C_flat[3]/2],
                       [C_flat[1]/2, C_flat[2], C_flat[4]/2],
                       [C_flat[3]/2, C_flat[4]/2, 1]])

    C_dash_inf = C_dash_inf/np.amax(C_dash_inf)

    return C_dash_inf


def estimate_H_from_conic(C_dash_inf):

    U, s_2, Vh = np.linalg.svd(C_dash_inf[0:2, 0:2])

    print("U = {}".format(U))
    print("s_2 = {}".format(s_2))

    # U1, S1, V1 = np.linalg.svd(C_dash_inf)
    # print("s1 = {}".format(S1))

    A = np.dot(np.dot(U , np.diag(np.sqrt(s_2))), U.T)

    H = np.zeros((3, 3))

    H[0:2, -1] = 0

    H[0:2, 0:2] = A

    v = np.dot(np.linalg.pinv(A), C_dash_inf[0:2, -1])

    H[-1, 0:2] = v.T

    H[-1, -1] = 1

    return H


def run_one_step(img_path, pts_1_on_line_1, pts_2_on_line_1, pts_1_on_line_2, pts_2_on_line_2, savefig_prefix=''):

    perp_lines_1 = get_line_equations(pts_1_on_line_1, pts_2_on_line_1)
    perp_lines_2 = get_line_equations(pts_1_on_line_2, pts_2_on_line_2)

    C_dash_inf = get_image_deg_conic(perp_lines_1, perp_lines_2)


    print(C_dash_inf)

    H = estimate_H_from_conic(C_dash_inf)  # img = H * world

    # Read img

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    #####################
    # without offset correction.

    world_img = np.zeros_like(img)


    mask = np.ones((img.shape[0], img.shape[1]))
    out = fit_image_in_target_space(img, world_img, mask, H)
    cv2.imwrite(savefig_prefix + "cv_1.jpg", out[:,:, (2,1,0)])

    plt.figure()
    plt.imshow(out)
    plt.title("Corrected image with Point point correspondence - NO OFFSET CORRECTION")
    plt.axis("off")
    plt.savefig(savefig_prefix + "_1.jpg")

    #####################
    # with offset correction.

    h, w, _ = img.shape

    # Figure out where the corners of image map to the world coordinates
    corners_img = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    H_inv = np.linalg.inv(H)

    t_one = np.ones((corners_img.shape[0], 1))
    t_out_pts = np.concatenate((corners_img, t_one), axis=1)
    world_crd_corners = np.matmul(H_inv, t_out_pts.T)
    world_crd_corners = world_crd_corners/world_crd_corners[-1, :]  # cols of [x1, y1, z1]

    min_crd = np.amin(world_crd_corners.T, axis=0)
    max_crd = np.max(world_crd_corners.T, axis=0)

    offset = min_crd.astype(np.int64)
    offset[2] = 0  # [x_offset, y_offset, 0]

    width_world = np.ceil(max_crd - min_crd)[0] + 1
    height_world = np.ceil(max_crd - min_crd)[1] + 1

    world_img = np.zeros((int(height_world), int(width_world), 3), dtype=np.int64)
    mask = np.ones((int(height_world), int(width_world)))

    out = fit_image_in_target_space(img, world_img, mask, H, offset)

    cv2.imwrite(savefig_prefix + "cv_2.jpg", out[:,:, (2,1,0)])

    plt.figure()
    plt.imshow(out)
    plt.axis("off")
    plt.title("Corrected image with One_step method")
    plt.savefig(savefig_prefix + "_2.jpg")
    plt.show()






if __name__ == "__main__" :

    a = [246, 1245]
    e = [2031, 402]
    d = [168, 1641]
    h = [2079, 1218]

    i = [1347, 1386]
    j = [1359, 1167]
    k = [1506, 1122]
    l = [1515, 1347]


    img_path = '/Users/aartighatkesar/Documents/homography_distortion_removal/Original_Images/1.jpg'

    pts_1_on_line_1 = [d, a, e, h, i]  # da perp ae; ae perp eh; eh perp hd; hd perp da; ik perp jl

    pts_2_on_line_1 = [a, e, h, d, k]

    pts_1_on_line_2 = [a, e, h, d, j]

    pts_2_on_line_2 = [e, h, d, a, l]

    run_one_step(img_path, pts_1_on_line_1, pts_2_on_line_1, pts_1_on_line_2, pts_2_on_line_2, savefig_prefix='onestep_building')


    ##########################

    a = [244, 70]
    b = [325, 82]
    c = [324, 265]
    d = [244, 268]

    e = [58, 133]
    f = [125, 137]
    g = [125, 229]
    h = [59, 230]

    img_path = '/Users/aartighatkesar/Documents/homography_distortion_removal/Original_Images/2.jpg'

    pts_1_on_line_1 = [a, b, c, d, e]  # ab perp bc; bc perp cd; cd perp da; da perp ab; eg perp fh

    pts_2_on_line_1 = [b, c, d, a, g]

    pts_1_on_line_2 = [b, c, d, a, f]

    pts_2_on_line_2 = [c, d, a, b, h]

    run_one_step(img_path, pts_1_on_line_1, pts_2_on_line_1, pts_1_on_line_2, pts_2_on_line_2,
                 savefig_prefix='onestep_painting')


