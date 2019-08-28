from estimate_homography import *


def get_line_equations(pts_1_on_line, pts_2_on_line):
    """
    Gets equation of a line. Can also be used to find intersection points of lines
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


def remove_projective(img, vanishing_line, savefig_prefix=''):
    """
    Function to get homography that maps vanishing line to l_inf. Corrects projective distortion
    :param vanishing_line:
    :return:
    """
    Hp = np.zeros((3, 3))
    Hp[-1, :] = vanishing_line

    Hp[0:2, 0:2] = np.eye(2)  # maps vanishing line to l_inf

    Hp = np.linalg.inv(Hp)  # for img = H * world, we need H which maps l_inf to vanishing line

    #####################
    # with offset correction.

    h, w, _ = img.shape

    # Figure out where the corners of image map to the world coordinates
    corners_img = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    Hp_inv = np.linalg.inv(Hp)

    t_one = np.ones((corners_img.shape[0], 1))
    t_out_pts = np.concatenate((corners_img, t_one), axis=1)
    world_crd_corners = np.matmul(Hp_inv, t_out_pts.T)
    world_crd_corners = world_crd_corners/world_crd_corners[-1, :]  # cols of [x1, y1, z1]

    min_crd = np.amin(world_crd_corners.T, axis=0)
    max_crd = np.max(world_crd_corners.T, axis=0)

    offset = min_crd.astype(np.int64)
    offset[2] = 0  # [x_offset, y_offset, 0]

    width_world = np.ceil(max_crd - min_crd)[0] + 1
    height_world = np.ceil(max_crd - min_crd)[1] + 1

    world_img = np.zeros((int(height_world), int(width_world), 3), dtype=np.int64)
    mask = np.ones((int(height_world), int(width_world)))

    out = fit_image_in_target_space(img, world_img, mask, Hp, offset)

    cv2.imwrite(savefig_prefix + "cv_2.jpg", out[:,:, (2,1,0)])

    plt.figure()
    plt.imshow(out)
    plt.axis("off")
    plt.title("Projective Corrected image with two_step method")
    plt.savefig(savefig_prefix + "_2.jpg")
    plt.show()

    return Hp

def build_two_step_eqns(perp_lines_1, perp_lines_2):
    """

    :param perp_lines_1: rows of  [l1, l2, l3]
    :param perp_lines_2:
    :return:
    """
    mat_A = np.zeros((perp_lines_1.shape[0], 2))
    mat_b = -1 * perp_lines_1[:, 1] * perp_lines_2[:, 1]  # -l2 * m2

    mat_A[:, 0] = perp_lines_1[:, 0] * perp_lines_2[:, 0]  # l1 * m1

    mat_A[:, 1] = perp_lines_1[:, 0] * perp_lines_2[:, 1] + perp_lines_1[:, 1] * perp_lines_2[:, 0]  # l1 * m2 + l2 * m1

    return mat_A, mat_b


def remove_affine_distortion(img, Hp, perp_l1_pts_1, perp_l1_pts_2, perp_l2_pts_1, perp_l2_pts_2, savefig_prefix=''):

    perp_l1_pts_1 = convert_to_homogenous_crd(perp_l1_pts_1)
    perp_l1_pts_2 = convert_to_homogenous_crd(perp_l1_pts_2)
    perp_l2_pts_1 = convert_to_homogenous_crd(perp_l2_pts_1)
    perp_l2_pts_2 = convert_to_homogenous_crd(perp_l2_pts_2)

    # Map the coordinates of lines in image plane to that in projective corrected world img

    Hp_inv = np.linalg.inv(Hp)

    tx_perp_l1_pts_1 = np.matmul(Hp_inv, perp_l1_pts_1.T).T
    tx_perp_l1_pts_2 = np.matmul(Hp_inv, perp_l1_pts_2.T).T
    tx_perp_l2_pts_1 = np.matmul(Hp_inv, perp_l2_pts_1.T).T
    tx_perp_l2_pts_2 = np.matmul(Hp_inv, perp_l2_pts_2.T).T

    perp_lines_1 = get_line_equations(tx_perp_l1_pts_1, tx_perp_l1_pts_2)  # Rows of line equations in homogenous crd
    perp_lines_2 = get_line_equations(tx_perp_l2_pts_1, tx_perp_l2_pts_2)  # Rows of line equations in homogenous crd


    mat_A, mat_b = build_two_step_eqns(perp_lines_1, perp_lines_2)

    s = np.matmul(np.linalg.pinv(mat_A), mat_b)

    S = np.array([[s[0], s[1]], [s[1], 1]])

    U, S_2, Vt = np.linalg.svd(S)

    A = np.dot(np.dot(U , np.diag(np.sqrt(S_2))), U.T)

    Ha = np.zeros((3, 3))

    Ha[0:2, 0:2] = A
    Ha[-1, -1] = 1

    Ha = Ha/np.amax(Ha)

    return Ha

def finalremoval_distortion(img, Ha, Hp, savefig_prefix=''):

    H = np.matmul(Hp, Ha)
    H = H/np.amax(H)

    world_img = np.zeros_like(img)

    mask = np.ones((img.shape[0], img.shape[1]))
    out = fit_image_in_target_space(img, world_img, mask, H)
    cv2.imwrite(savefig_prefix + "cv_1.jpg", out[:,:, (2,1,0)])

    plt.figure()
    plt.imshow(out)
    plt.title("Corrected image with two step correspondence - NO OFFSET CORRECTION")
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
    plt.title("Final Corrected image with two_step method")
    plt.savefig(savefig_prefix + "_2.jpg")
    plt.show()


def run_two_step(img_path, pts_1_pl1, pts_2_pl1, pts_1_pl2, pts_2_pl2,
                 perp_l1_pts_1, perp_l1_pts_2, perp_l2_pts_1, perp_l2_pts_2,
                 savefig_prefix=''):

    parallel_lns_1 = get_line_equations(pts_1_pl1, pts_2_pl1)
    parallel_lns_2 = get_line_equations(pts_1_pl2, pts_2_pl2)


    vp1 = np.cross(parallel_lns_1[0:1, :], parallel_lns_1[1:2, :])
    vp1 = vp1/ vp1[:, -1:]

    vp2 = np.cross(parallel_lns_2[0:1, :], parallel_lns_2[1:2, :])
    vp2 = vp2 / vp2[:, -1:]

    vanishing_line = get_line_equations(vp1, vp2)

    print("Vanishing line :{}".format(vanishing_line))

    # Read img
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    Hp = remove_projective(img, vanishing_line, savefig_prefix=savefig_prefix+'projective_removal')

    Ha = remove_affine_distortion(img, Hp, perp_l1_pts_1, perp_l1_pts_2, perp_l2_pts_1, perp_l2_pts_2)

    finalremoval_distortion(img, Hp=Hp, Ha=Ha, savefig_prefix=savefig_prefix)


if __name__ == "__main__":
    a = [246, 1245]
    e = [2031, 402]
    d = [168, 1641]
    h = [2079, 1218]

    i = [1347, 1386]
    j = [1359, 1167]
    k = [1506, 1122]
    l = [1515, 1347]


    img_path = '/Users/aartighatkesar/Documents/homography_distortion_removal/Original_Images/1.jpg'

    pts_1_pl1 = [a, d]  # ad // eh
    pts_2_pl1 = [e, h]

    pts_1_pl2 = [a, e]  # ae //dh
    pts_2_pl2 = [d, h]

    perp_l1_pts_1 = [a, j]  # ae perp da; jl perp ik
    perp_l1_pts_2 = [e, l]
    perp_l2_pts_1 = [d, i]
    perp_l2_pts_2 = [a, k]



    run_two_step(img_path, pts_1_pl1, pts_2_pl1, pts_1_pl2, pts_2_pl2,
                 perp_l1_pts_1, perp_l1_pts_2, perp_l2_pts_1, perp_l2_pts_2,
                 savefig_prefix='two_step_building')



    #################################

    a = [244, 70]
    b = [325, 82]
    c = [324, 265]
    d = [244, 268]

    e = [58, 133]
    f = [125, 137]
    g = [125, 229]
    h = [59, 230]


    img_path = '/Users/aartighatkesar/Documents/homography_distortion_removal/Original_Images/2.jpg'

    pts_1_pl1 = [a, d]  # ad // bc ;
    pts_2_pl1 = [b, c]

    pts_1_pl2 = [a, b]  # ab // dc
    pts_2_pl2 = [d, c]

    perp_l1_pts_1 = [a, e]  # ab perp bc; eg perp fh
    perp_l1_pts_2 = [b, g]
    perp_l2_pts_1 = [b, f]
    perp_l2_pts_2 = [c, h]

    run_two_step(img_path, pts_1_pl1, pts_2_pl1, pts_1_pl2, pts_2_pl2,
                 perp_l1_pts_1, perp_l1_pts_2, perp_l2_pts_1, perp_l2_pts_2,
                 savefig_prefix='two_step_painting')