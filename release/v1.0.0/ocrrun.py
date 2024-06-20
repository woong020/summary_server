import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
import scipy.optimize
import paddleocr
from paddleocr import PaddleOCR, draw_ocr



# for some reason pylint complains about cv2 members being undefined :(
# pylint: disable=E1101
save_count = 0

PAGE_MARGIN_X = 13       # reduced px to ignore near L/R edge #줄일수록 범위가 넓어짐 기본 50
PAGE_MARGIN_Y = 20       # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 600         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 3      # min reduced px width of detected text contour 기본 15
TEXT_MIN_HEIGHT = 1.5      # min reduced px height of detected text contour 기본 2
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio 기본 1.5
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour 기본 10

EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
FOCAL_LENGTH = 1.2       # normalized focal length of camera

DEBUG_LEVEL = 0          # 0=none, 1=some, 2=lots, 3=all
DEBUG_OUTPUT = 'file'    # file, screen, both

WINDOW_NAME = 'Dewarp'   # Window name for visualization

# nice color palette for visualizing contours, etc.
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]

# PaddleOCR 모델을 로드합니다.
ocr = PaddleOCR(lang="korean", use_gpu=True)
#PaddleOCR(show_log = False)

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)


def debug_show(name, step, text, display):

    if DEBUG_OUTPUT != 'screen':
        filetext = text.replace(' ', '_')
        outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != 'file':

        image = display.copy()
        height = image.shape[0]

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(image, text, (16, height-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)

        while cv2.waitKey(5) < 0:
            pass


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem


def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
    return (pts - offset) * scl


def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def fltp(point):
    return tuple(point.astype(int).flatten())


def draw_correspondences(img, dstpoints, projpts):
    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 1, cv2.LINE_AA)

    return display


def get_default_params(corners, ycoords, xcoords):
    # page width and height
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (page_width, page_height)

    # our initial guess for the cubic has no slope
    cubic_slopes = [0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])

    # estimate rotation and translation from four 2D-to-3D point
    # correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))

    span_counts = [len(xc) for xc in xcoords]

    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) +
                       tuple(xcoords))

    return rough_dims, span_counts, params


def project_xy(xy_coords, pvec):
    alpha, beta = tuple(pvec[CUBIC_IDX])

    poly = np.array([
        alpha + beta,
        -2*alpha - beta,
        alpha,
        0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        K, np.zeros(5))

    return image_points


def project_keypoints(pvec, keypoint_index):
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0

    return project_xy(xy_coords, pvec)


def resize_to_screen(src, maxw=1280, maxh=700, copy=False):
    height, width = src.shape[:2]

    scl_x = float(width)/maxw
    scl_y = float(height)/maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0/scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)

def text_mask(small):
    global DEBUG_LEVEL
    global TEXT_MIN_WIDTH
    global TEXT_MIN_HEIGHT
    global TEXT_MAX_THICKNESS

    tmp_debug_level = DEBUG_LEVEL
    DEBUG_LEVEL = 0
    tmp_text_min_width = TEXT_MIN_WIDTH
    TEXT_MIN_WIDTH = 5
    tmp_text_min_height = TEXT_MIN_HEIGHT
    TEXT_MIN_HEIGHT = 1
    tmp_text_max_thickness = TEXT_MAX_THICKNESS
    TEXT_MAX_THICKNESS = 20

    name = 'mask_temp'
    spans = []
    height, width = small.shape[:2]
    pagemask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(pagemask, (0, 0), (width, height), (255, 255, 255), -1)

    # 윤곽선 정보 검출
    cinfo_list = get_contours(name, small, pagemask, 'text')
    # 윤곽선을 길게 합침
    spans_text = assemble_spans(name, small, pagemask, cinfo_list)

    # 텍스트가 없어서 span 갯수가 부족시 line을 디텍팅 하도록 설계
    if len(spans_text) < 3:
        print ('  detecting lines because only', len(spans_text), 'text spans')
        cinfo_list = get_contours(name, small, pagemask, 'line')
        spans2 = assemble_spans(name, small, pagemask, cinfo_list)
        if len(spans2) > len(spans_text):
            spans_text = spans2

    # 윤곽선 각각의 넓이와 y축 위치 구하기.
    span_with_min = []
    span_with_max = []
    span_height_min = []
    span_height_max = []
    for i, span in enumerate(spans_text):
        # 한줄 빼오기
        contours = [cinfo.contour for cinfo in span]
        # 빼온거에서 젤 왼쪽과 오른쪽 을 빼서 길이 재고 y축 값 구하기.
        contours_min = np.argmin(contours[0], axis = 0)
        contours_max = np.argmax(contours[0], axis = 0)
        span_with_min.append(contours[0][contours_min[0][0]][0][0])
        span_with_max.append(contours[0][contours_max[0][0]][0][0])
        span_height_min.append(contours[0][contours_min[0][1]][0][1])
        span_height_max.append(contours[0][contours_max[0][1]][0][1])

    DEBUG_LEVEL = tmp_debug_level
    TEXT_MIN_WIDTH = tmp_text_min_width
    TEXT_MIN_HEIGHT = tmp_text_min_height
    TEXT_MAX_THICKNESS = tmp_text_max_thickness

    # 리턴 값은 x축 최소값, x축 최대값, y축 최소값, y축 최대값
    return np.min(span_with_min), np.max(span_with_max), np.min(span_height_min), np.max(span_height_max)


def get_page_extents(small):
    height, width = small.shape[:2]

    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width-PAGE_MARGIN_X
    ymax = height-PAGE_MARGIN_Y
    
    
    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


def get_mask(name, small, pagemask, masktype):
    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    if masktype == 'text':

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     25)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.1, 'thresholded', mask)

        mask = cv2.dilate(mask, box(9, 1))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.2, 'dilated', mask)

        mask = cv2.erode(mask, box(1, 3))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.3, 'eroded', mask)

    else:
        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     7)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(8, 2))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.6, 'dilated', mask)

    return np.minimum(mask, pagemask)


def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


def angle_dist(angle_b, angle_a):
    diff = angle_b - angle_a

    while diff > np.pi:
        diff -= 2*np.pi

    while diff < -np.pi:
        diff += 2*np.pi

    return np.abs(diff)


def blob_mean_and_tangent(contour):
    moments = cv2.moments(contour)

    area = moments['m00']

    mean_x = moments['m10'] / area
    mean_y = moments['m01'] / area

    moments_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area

    _, svd_u, _ = cv2.SVDecomp(moments_matrix)

    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()

    return center, tangent


class ContourInfo(object):

    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = blob_mean_and_tangent(contour)

        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten()-self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def generate_candidate_edge(cinfo_a, cinfo_b):
    # we want a left of b (so a's successor will be b and b's
    # predecessor will be a) make sure right endpoint of b is to the
    # right of left endpoint of a.
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp

    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi

    # we want the largest overlap in x to be small
    x_overlap = max(x_overlap_a, x_overlap_b)

    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

    if (dist > EDGE_MAX_LENGTH or
            x_overlap > EDGE_MAX_OVERLAP or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle*EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)


def make_tight_mask(contour, xmin, ymin, width, height):
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)

    return tight_mask


def get_contours(name, small, pagemask, masktype):
    mask = get_mask(name, small, pagemask, masktype)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_out = []

    for contour in contours:

        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT*height):
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    if DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)

    return contours_out


def assemble_spans(name, small, pagemask, cinfo_list):

    # sort list
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])

    # generate all candidate edges
    candidate_edges = []

    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # sort candidate edges by score (lower is better)
    candidate_edges.sort()

    # for each candidate edge
    for _, cinfo_a, cinfo_b in candidate_edges:
        # if left and right are unassigned, join them
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # generate list of spans as output
    spans = []

    # until we have removed everything from the list
    while cinfo_list:

        # get the first on the list
        cinfo = cinfo_list[0]

        # keep following predecessors until none exists
        while cinfo.pred:
            cinfo = cinfo.pred

        # start a new span
        cur_span = []

        width = 0.0

        # follow successors til end of span
        while cinfo:
            # remove from list (sadly making this loop *also* O(n^2)
            cinfo_list.remove(cinfo)
            # add to span
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            # set successor
            cinfo = cinfo.succ

        # add if long enough
        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    if DEBUG_LEVEL >= 2:
        visualize_spans(name, small, pagemask, spans)

    return spans


def sample_spans(shape, spans):
    span_points = []

    for span in spans:

        contour_points = []

        for cinfo in span:

            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = totals / cinfo.mask.sum(axis=0)

            xmin, ymin = cinfo.rect[:2]

            step = SPAN_PX_PER_STEP
            start = ((len(means)-1) % step) / 2

            contour_points += [(x+xmin, means[x]+ymin)
                               for x in range(int(start), len(means), int(step))] #########################################

        contour_points = np.array(contour_points,
                                  dtype=np.float32).reshape((-1, 1, 2))

        contour_points = pix2norm(shape, contour_points)

        span_points.append(contour_points)

    return span_points


def keypoints_from_samples(name, small, pagemask, page_outline,
                           span_points):
    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:

        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)

        weight = np.linalg.norm(points[-1] - points[0])

        all_evecs += evec * weight
        all_weights += weight

    evec = all_evecs / all_weights

    x_dir = evec.flatten()

    if x_dir[0] < 0:
        x_dir = -x_dir

    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)

    px0 = px_coords.min()
    px1 = px_coords.max()

    py0 = py_coords.min()
    py1 = py_coords.max()

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir

    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []

    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    if DEBUG_LEVEL >= 2:
        visualize_span_points(name, small, span_points, corners)

    return corners, np.array(ycoords), xcoords


def visualize_contours(name, small, cinfo_list):
    regions = np.zeros_like(small)

    for j, cinfo in enumerate(cinfo_list):

        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)

    for j, cinfo in enumerate(cinfo_list):
        color = CCOLORS[j % len(CCOLORS)]
        color = tuple([c/4 for c in color])

        cv2.circle(display, fltp(cinfo.center), 3,
                   (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    debug_show(name, 1, 'contours', display)


def visualize_spans(name, small, pagemask, spans):
    regions = np.zeros_like(small)

    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        cv2.drawContours(regions, contours, -1,
                         CCOLORS[i*3 % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (display[mask]/2) + (regions[mask]/2)
    display[pagemask == 0] /= 4

    debug_show(name, 2, 'spans', display)


def visualize_span_points(name, small, span_points, corners):
    display = small.copy()

    for i, points in enumerate(span_points):

        points = norm2pix(small.shape, points, False)

        mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
                                          None,
                                          maxComponents=1)

        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())

        point0 = mean + small_evec * (dps.min()-dpm)
        point1 = mean + small_evec * (dps.max()-dpm)

        for point in points:
            cv2.circle(display, fltp(point), 3,
                       CCOLORS[i % len(CCOLORS)], -1, cv2.LINE_AA)

        cv2.line(display, fltp(point0), fltp(point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.polylines(display, [norm2pix(small.shape, corners, True)],
                  True, (255, 255, 255))

    debug_show(name, 3, 'span points', display)


def imgsize(img):
    height, width = img.shape[:2]
    return '{}x{}'.format(width, height)


def make_keypoint_index(span_counts):
    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts+1, 2), dtype=int)
    start = 1

    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start+end, 1] = 8+i
        start = end

    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans

    return keypoint_index


def optimize_params(name, small, dstpoints, span_counts, params):
    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts)**2)

    print( '  initial objective is', objective(params))

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, 'keypoints before', display)

    print( '  optimizing', len(params), 'parameters...')
    start = datetime.datetime.now()
    res = scipy.optimize.minimize(objective, params,
                                  method='Powell')
    end = datetime.datetime.now()
    print( '  optimization took', round((end-start).total_seconds(), 2), 'sec.')
    print( '  final objective is', res.fun)
    params = res.x

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 5, 'keypoints after', display)

    return params


def get_page_dims(corners, rough_dims, params):
    dst_br = corners[2].flatten()

    dims = np.array(rough_dims)

    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten())**2)

    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x

    print( '  got page dims', dims[0], 'x', dims[1])

    return dims


def remap_image(name, img, small, page_dims, params, output_dir):
        height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
        height = round_nearest_multiple(height, REMAP_DECIMATE)

        width = round_nearest_multiple(height * page_dims[0] / page_dims[1],
                                    REMAP_DECIMATE)

        print( '  output will be {}x{}'.format(width, height))

        height_small = height / REMAP_DECIMATE
        width_small = width / REMAP_DECIMATE

        page_x_range = np.linspace(0, page_dims[0], int(width_small))   ############################
        page_y_range = np.linspace(0, page_dims[1], int(height_small)) #############################

        page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

        page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                    page_y_coords.flatten().reshape((-1, 1))))

        page_xy_coords = page_xy_coords.astype(np.float32)

        image_points = project_xy(page_xy_coords, params)
        image_points = norm2pix(img.shape, image_points, False)

        image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
        image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

        image_x_coords = cv2.resize(image_x_coords, (width, height),
                                    interpolation=cv2.INTER_LANCZOS4) ## INTER_CUBIC > cv2.INTER_LANCZOS4

        image_y_coords = cv2.resize(image_y_coords, (width, height),
                                    interpolation=cv2.INTER_LANCZOS4) ## INTER_CUBIC > cv2.INTER_LANCZOS4

        remapped = cv2.remap(img, image_x_coords, image_y_coords,
                            cv2.INTER_LANCZOS4,                 ## INTER_CUBIC > cv2.INTER_LANCZOS4
                            None, cv2.BORDER_REPLICATE)


        pil_image = remapped

        if DEBUG_LEVEL >= 1:
            try:
                # Save the image
                pil_image.save(threshfile, dpi=(OUTPUT_DPI, OUTPUT_DPI))
                print('Image saved to:', threshfile)
            except Exception as e:
                print('Error saving image:', e)

            height = small.shape[0]
            width = int(round(height * float(remapped.shape[1])/remapped.shape[0])) #remapped > thresh
            display = cv2.resize(remapped, (width, height), #remapped > thresh
                                interpolation=cv2.INTER_AREA)
            debug_show(name, 6, 'output', display)

        #return threshfile
        return pil_image

# 이미지 백그라운드 삭제
def crop_image(image, scan_size, left, right, top, bottom): #left~ bottom 변수 삭제 가능
     # 이미지의 높이와 너비를 가져옵니다
    height_1, width, _ = image.shape

    if scan_size == 1:
        left = 1100 ## 조정 필요
        right = 1050 ## 조정 필요
        top = 350  ## 조정 필요
        bottom = 100  ## 조정 필요
    elif scan_size == 2:
        left = 1100 ## 조정 필요
        right = 750 ## 조정 필요
        top = 400  ## 조정 필요
        bottom = 0  ## 조정 필요
    elif scan_size == 3:
        left = 1250 ## 조정 필요
        right = 1050 ## 조정 필요
        top = 350  ## 조정 필요
        bottom = 400  ## 조정 필요
    

    # 자를 영역을 설정합니다
    start_x = left
    end_x = width - right
    start_y = top
    end_y = height_1 - bottom

    # 이미지를 자릅니다
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

#책 이미지 반갈
def cut_half(image):

    # 이미지를 RGB 형식으로 변환
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height_2, width = image.shape[:2]

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 에지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 선을 추출
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=5)

    tmp_candy = []
    tmp_abs = []

    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            gapY = np.abs(y2 - y1)
            gapX = np.abs(x2 - x1)
            # 선의 x축의 차이가 5이하 y축의 차이가 10이상이면 세로줄로 판별
            if gapX < 5 and gapY > 50:
                tmp_candy.append(x1)
                tmp_abs.append(np.abs(x1 - width // 2))

    # 가장 중심에 가까운 선으로 이미지를 좌우로 자름
    split_idx = np.argmin(tmp_abs)
    left_img = img_rgb[:, :tmp_candy[split_idx]]
    right_img = img_rgb[:, tmp_candy[split_idx]:]

    return left_img, right_img


#이미지 자르기 및 저장
def crop_and_cut(image_path, scan_size, left, right, top, bottom): #left~ bottom 변수 삭제 가능
    # 이미지를 읽어옵니다
    image = cv2.imread(image_path)

    # 이미지를 자릅니다
    cropped_image = crop_image(image, scan_size, left, right, top, bottom) #left~ bottom 변수 삭제 가능

    # 왼쪽 이미지와 오른쪽 이미지를 얻습니다
    left_img, right_img = cut_half(cropped_image)
        
    return left_img, right_img


# 이미지 하단 영역 삭제
def crop_image_bottom(image_path, pil_image, d_top, d_bottom):
     # 이미지의 높이와 너비를 가져옵니다
    height_1, width, _ = pil_image.shape

    # 자를 영역을 설정합니다
    start_x = 0
    end_x = width - 0
    start_y = d_top
    end_y = height_1 - d_bottom

    # 이미지를 자릅니다
    cropped_image_bottom = pil_image[start_y:end_y, start_x:end_x]

    return cropped_image_bottom


def ocr_apply(pil_image, output_dir):
    global save_count

    # 이미지에서 텍스트를 추출합니다.
    result = ocr.ocr(pil_image, cls=True)

    # 추출된 텍스트를 한 문장으로 이어서 출력합니다.
    text = ''
    if result:  # Check if result is not empty or None
        for line in result:
            if line:  # Check if line is not None
                try:
                    text += ' '.join([word[1][0] for word in line]) + ' '
                except TypeError as e:
                    print(f"TypeError occurred: {e}")
                    continue

    # 파일 이름이 중복되지 않도록 확인하고 저장
    base_filename = "text"
    extension = ".txt"
    output_filename = os.path.join(output_dir, f"{base_filename}{extension}")

    # 파일이 존재하지 않으면 새로운 파일 생성
    if not os.path.exists(output_filename):
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f'Extracted text saved to: {output_filename}')
        save_count = 1  # 파일 생성 후 첫 번째 저장으로 설정
    else:
        if save_count == 0:
            # 초기 save_count 설정
            with open(output_filename, "r", encoding="utf-8") as f:
                content = f.read()
                if content:
                    save_count = content.count("Extracted text saved to") + content.count(
                        "Extracted text appended to") + content.count("Extracted text overwritten to")

        # 2회 저장 후 save_count를 리셋
        if save_count % 2 == 0:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f'Extracted text overwritten to: {output_filename}')
        else:
            with open(output_filename, "a", encoding="utf-8") as f:
                f.write(text)
            print(f'Extracted text appended to: {output_filename}')

        save_count = (save_count + 1) % 2  # save_count를 2회마다 리셋

# server.py 에서 호출되는 메소드. scan_size를 파라미터로 사용
def main(scan_size):
    
    import sys
    import os
    # 경로 재생성 방지
    sys.argv = [sys.argv[0]]

    ##### 주요 파라미터 #####
    # 15 - 16줄: 텍스트 박스 영역 margin 
    # 916 - 930줄 left-bottom: 이미지 배경 cut size 
    # 116x줄 image_path: 사진 찍은 이미지 경로
    # 116x줄 scan_size: scan size 클릭[size 선택 부분]
    # 117x줄 d_top, d_bottom: 이미지 상하단 cut size 

    # 이미지 자르기 및 반으로 나누기
    image_path = r"received_files/scan.jpg"

    left = 0 ## default #left~ bottom 변수 삭제 가능
    right = 0 ## default #left~ bottom 변수 삭제 가능
    top = 0  ## default  #left~ bottom 변수 삭제 가능
    bottom = 0  ## default #left~ bottom 변수 삭제 가능

    if scan_size == 1:
        d_top = 0
        d_bottom = 300
    elif scan_size == 2:
        d_top = 270
        d_bottom = 0
    elif scan_size == 3:
        d_top = 200
        d_bottom = 175  # 300 ## 조정 필요 #size 2는 따로주기


    # crop_and_cut 함수 호출 후 반환 값을 받음
    left_img, right_img = crop_and_cut(image_path, scan_size, left, right, top, bottom) # left ~ bottom 변수 삭제 가능

    # 임시 파일 경로 생성
    temp_left_path = "left_temp.jpg"
    temp_right_path = "right_temp.jpg"

    # 임시 파일로 저장
    cv2.imwrite(temp_left_path, left_img)
    cv2.imwrite(temp_right_path, right_img)

    # 임시 파일 경로를 sys.argv에 추가
    sys.argv.extend([temp_left_path, temp_right_path])

    # ==================================================================

    # ① 파라미터 값은 이미지 파일 하나 이상이 들어와야 함
    if len(sys.argv) < 2:
        print( 'usage:', sys.argv[0], 'IMAGE1 [IMAGE2 ...]')
        sys.exit(0)

    # ② 결과를 파일이 아닌 스크린으로 보여줄때 윈도우 창 이름 설정
    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != 'file':
        cv2.namedWindow(WINDOW_NAME)
    
    outfiles = []

    # ③ input 파일 개수만큼 for문을 돌림 (한개일땐 한번만...)
    for imgfile in sys.argv[1:]:
        # ④ 파일 이름으로 openCV에서 파일을 읽음
        img = cv2.imread(imgfile)
        small = resize_to_screen(img)
        basename = os.path.basename(imgfile)
        name, _ = os.path.splitext(basename)
        output_dir = os.path.dirname(image_path)  # 이미지 파일의 디렉토리
        #output_dir = image_path

        print( 'loaded', basename, 'with size', imgsize(img))
        print( 'and resized to', imgsize(small))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.0, 'original', small)

        pagemask, page_outline = get_page_extents(small)

        cinfo_list = get_contours(name, small, pagemask, 'text')
        spans = assemble_spans(name, small, pagemask, cinfo_list)

        if len(spans) < 3:
            print( '  detecting lines because only', len(spans), 'text spans')
            cinfo_list = get_contours(name, small, pagemask, 'line')
            spans2 = assemble_spans(name, small, pagemask, cinfo_list)
            if len(spans2) > len(spans):
                spans = spans2

        if len(spans) < 1:
            print( 'skipping', name, 'because only', len(spans), 'spans')
            continue

        span_points = sample_spans(small.shape, spans)

        print( '  got', len(spans), 'spans')
        print( 'with', sum([len(pts) for pts in span_points]), 'points.')

        corners, ycoords, xcoords = keypoints_from_samples(name, small,
                                                           pagemask,
                                                           page_outline,
                                                           span_points)

        rough_dims, span_counts, params = get_default_params(corners,
                                                             ycoords, xcoords)

        dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
                              tuple(span_points))

        params = optimize_params(name, small,
                                 dstpoints,
                                 span_counts, params)

        page_dims = get_page_dims(corners, rough_dims, params)

        pil_image = remap_image(name, img, small, page_dims, params, output_dir)
        outfiles.append(pil_image)
        
        # image 하단영역 삭제
        cropped_image_bottom = crop_image_bottom(image_path, pil_image, d_top, d_bottom)
        # Apply OCR after remapping the image
        ocr_apply(cropped_image_bottom, output_dir)

        print("=====ocr run complete=====")
    
    # 임시 파일 삭제 후 return 반환
    os.remove(temp_left_path)
    os.remove(temp_right_path)
    return True
